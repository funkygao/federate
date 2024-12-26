package main

import (
	"fmt"
	"sync"
	"time"
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
	bookKeeper  BookKeeper
	topic       *Topic
	name        string
	subType     SubscriptionType
	ackMessages map[MessageID]bool
	mu          sync.Mutex
	cursor      MessageID
}

func NewInMemorySubscription(bk BookKeeper, topic *Topic, name string, subType SubscriptionType) *InMemorySubscription {
	return &InMemorySubscription{
		bookKeeper:  bk,
		topic:       topic,
		name:        name,
		subType:     subType,
		ackMessages: make(map[MessageID]bool),
		cursor: MessageID{
			PartitionID: 0,
			LedgerID:    0,
			EntryID:     0,
		},
	}
}

func (s *InMemorySubscription) Ack(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.ackMessages[msgID]; !exists {
		return fmt.Errorf("message %+v not found", msgID)
	}

	delete(s.ackMessages, msgID)
	s.cursor = msgID

	return nil
}

func (s *InMemorySubscription) Fetch() (Message, error) {
	partition := s.topic.GetPartition(s.cursor.PartitionID)

	for {
		if err := s.initializeCursorIfNeeded(partition); err != nil {
			continue
		}

		ledger, err := s.bookKeeper.OpenLedger(s.cursor.LedgerID)
		if err != nil {
			return Message{}, fmt.Errorf("failed to open ledger: %v", err)
		}

		msg, found, err := s.tryReadNextEntry(ledger)
		if err != nil {
			return Message{}, err
		}
		if found {
			return msg, nil
		}

		if s.moveToNextLedgerIfAvailable(partition) {
			continue
		}

		time.Sleep(100 * time.Millisecond)
	}
}

func (s *InMemorySubscription) initializeCursorIfNeeded(partition *Partition) error {
	if s.cursor.LedgerID == 0 {
		if len(partition.Ledgers) == 0 {
			time.Sleep(100 * time.Millisecond)
			return fmt.Errorf("no ledgers available")
		}
		s.cursor.LedgerID = partition.Ledgers[0]
		s.cursor.EntryID = 0
	}
	return nil
}

func (s *InMemorySubscription) tryReadNextEntry(ledger Ledger) (Message, bool, error) {
	lastConfirmed := ledger.GetLastAddConfirmed()
	nextEntryID := s.cursor.EntryID + 1

	if nextEntryID <= lastConfirmed {
		payload, err := ledger.ReadEntry(nextEntryID)
		if err != nil {
			return Message{}, false, fmt.Errorf("failed to read entry: %v", err)
		}

		msg := Message{
			ID: MessageID{
				PartitionID: s.cursor.PartitionID,
				LedgerID:    s.cursor.LedgerID,
				EntryID:     nextEntryID,
			},
			Content: payload,
		}

		s.ackMessages[msg.ID] = true
		s.cursor.EntryID = nextEntryID

		return msg, true, nil
	}

	return Message{}, false, nil
}

func (s *InMemorySubscription) moveToNextLedgerIfAvailable(partition *Partition) bool {
	ledgerIndex := indexOfLedger(partition.Ledgers, s.cursor.LedgerID)
	if ledgerIndex+1 < len(partition.Ledgers) {
		s.cursor.LedgerID = partition.Ledgers[ledgerIndex+1]
		s.cursor.EntryID = 0
		return true
	}
	return false
}

func indexOfLedger(ledgers []LedgerID, ledgerID LedgerID) int {
	for i, id := range ledgers {
		if id == ledgerID {
			return i
		}
	}
	return -1
}
