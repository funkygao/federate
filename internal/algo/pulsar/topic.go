package main

import (
	"sync"
	"time"
)

type TopicManager struct {
	bookKeeper       BookKeeperService
	currentPartition PartitionID
	currentSegment   TimeSegmentID
	currentLedger    LedgerID
	nextEntryID      EntryID
	subscriptions    map[string]Subscription
	delayedMsgs      *DelayQueue
	mu               sync.Mutex
}

func NewTopicManager(bk BookKeeperService) *TopicManager {
	tm := &TopicManager{
		bookKeeper:       bk,
		currentPartition: 0,
		currentSegment:   0,
		currentLedger:    0,
		nextEntryID:      0,
		subscriptions:    make(map[string]Subscription),
		delayedMsgs:      NewDelayQueue(),
	}
	go tm.processDelayedMessages()
	return tm
}

func (tm *TopicManager) Publish(msg Message) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	entryID, err := tm.bookKeeper.AddEntry(tm.currentPartition, tm.currentSegment, tm.currentLedger, msg.Content)
	if err != nil {
		// 如果添加失败，可能需要创建新的 ledger
		newLedgerID, err := tm.bookKeeper.CreateLedger(tm.currentPartition, tm.currentSegment)
		if err != nil {
			return err
		}
		tm.currentLedger = newLedgerID
		entryID, err = tm.bookKeeper.AddEntry(tm.currentPartition, tm.currentSegment, tm.currentLedger, msg.Content)
		if err != nil {
			return err
		}
	}

	msg.ID = MessageID{
		LedgerID:      tm.currentLedger,
		EntryID:       entryID,
		PartitionID:   tm.currentPartition,
		TimeSegmentID: tm.currentSegment,
	}

	if msg.Delay > 0 {
		tm.delayedMsgs.Add(msg, time.Now().Add(msg.Delay))
	} else {
		tm.deliverMessage(msg)
	}
	return nil
}

func (tm *TopicManager) deliverMessage(msg Message) {
	for _, sub := range tm.subscriptions {
		sub.(*PulsarSubscription).NotifyNewMessage(msg)
	}
}

func (tm *TopicManager) processDelayedMessages() {
	for {
		msg, ok := tm.delayedMsgs.Poll()
		if ok {
			tm.deliverMessage(msg)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func (tm *TopicManager) Subscribe(name string, subType SubscriptionType) (Subscription, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if sub, exists := tm.subscriptions[name]; exists {
		return sub, nil
	}

	sub := NewPulsarSubscription(tm.bookKeeper, subType)
	tm.subscriptions[name] = sub
	return sub, nil
}
