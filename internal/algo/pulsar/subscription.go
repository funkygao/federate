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

// Subscription: Manages message acknowledgment and cursor positions.
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

	cursor MessageID
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
	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		// If the cursor's LedgerID is 0 (uninitialized), try to initialize it
		if s.cursor.LedgerID == 0 {
			partition := s.topic.GetPartition(s.cursor.PartitionID)
			if len(partition.Ledgers) == 0 {
				// No ledgers available yet, wait and retry
				s.mu.Unlock()
				time.Sleep(100 * time.Millisecond)
				s.mu.Lock()
				continue
			}
			// Set the cursor to the first ledger and EntryID to 0
			s.cursor.LedgerID = partition.Ledgers[0]
			s.cursor.EntryID = 0
		}

		// Open the ledger
		ledger, err := s.bookKeeper.OpenLedger(s.cursor.LedgerID)
		if err != nil {
			// Ledger not found, reset cursor and wait
			s.cursor.LedgerID = 0
			s.cursor.EntryID = 0
			s.mu.Unlock()
			time.Sleep(100 * time.Millisecond)
			s.mu.Lock()
			continue
		}

		// Get the last confirmed entry ID
		lastConfirmed := ledger.GetLastAddConfirmed()

		// Calculate the next entry to read
		nextEntryID := s.cursor.EntryID + 1

		if nextEntryID > lastConfirmed {
			// No new entries yet, wait and retry
			s.mu.Unlock()
			time.Sleep(100 * time.Millisecond)
			s.mu.Lock()
			continue
		}

		// Read the entry
		payload, err := ledger.ReadEntry(nextEntryID)
		if err != nil {
			// If entry not found due to ledger rollover, move to next ledger
			partition := s.topic.GetPartition(s.cursor.PartitionID)
			ledgerIndex := indexOfLedger(partition.Ledgers, s.cursor.LedgerID)
			if ledgerIndex+1 < len(partition.Ledgers) {
				// Move to the next ledger
				s.cursor.LedgerID = partition.Ledgers[ledgerIndex+1]
				s.cursor.EntryID = 0
				continue
			} else {
				// No new ledger yet, wait and retry
				s.mu.Unlock()
				time.Sleep(100 * time.Millisecond)
				s.mu.Lock()
				continue
			}
		}

		// Successfully read an entry, update the cursor
		s.cursor.EntryID = nextEntryID

		msg := Message{
			ID: MessageID{
				PartitionID: s.cursor.PartitionID,
				LedgerID:    s.cursor.LedgerID,
				EntryID:     s.cursor.EntryID,
			},
			Content: payload,
		}

		return msg, nil
	}
}
