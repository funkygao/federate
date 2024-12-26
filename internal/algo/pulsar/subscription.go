package main

import (
	"fmt"
	"sync"
)

type SubscriptionType int

const (
	Exclusive SubscriptionType = iota
	Shared
	Failover
)

type Subscription interface {
	Fetch() (Message, error)
	Ack(msgID MessageID) error
}

type InMemorySubscription struct {
	bookKeeper BookKeeper

	topic   *Topic
	name    string
	subType SubscriptionType

	ackMessages map[MessageID]bool
	mu          sync.Mutex

	cursor MessageID // TODO for delay msg, 1 cursor?
}

func NewInMemorySubscription(bk BookKeeper, topic *Topic, name string, subType SubscriptionType) *InMemorySubscription {
	return &InMemorySubscription{
		bookKeeper:  bk,
		topic:       topic,
		name:        name,
		subType:     subType,
		ackMessages: make(map[MessageID]bool),
	}
}

func (s *InMemorySubscription) Ack(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.ackMessages[msgID]; !exists {
		return fmt.Errorf("message not found")
	}

	delete(s.ackMessages, msgID)
	s.cursor = msgID

	return nil
}

func (s *InMemorySubscription) Fetch() (Message, error) {
	// 从 BookKeeper 获取下一条消息
	ledger, err := s.bookKeeper.OpenLedger(s.cursor.LedgerID)
	if err != nil {
		return Message{}, err
	}

	entry, err := ledger.ReadEntry(s.cursor.EntryID + 1)
	if err != nil {
		return Message{}, err
	}

	msg := Message{
		ID: MessageID{
			PartitionID:   s.cursor.PartitionID,
			TimeSegmentID: s.cursor.TimeSegmentID,
			LedgerID:      s.cursor.LedgerID,
			EntryID:       s.cursor.EntryID + 1,
		},
		Content: entry,
	}

	// 更新游标
	s.cursor = msg.ID

	return msg, nil
}
