package main

import (
	"fmt"
	"sync"
)

type LedgerOption struct {
	EnsembleSize int
	WriteQuorum  int
	AckQuorum    int
}

// Ledger: Represents a sequence of entries.
type Ledger interface {
	AddEntry(Payload, LedgerOption) (EntryID, error)
	ReadEntry(EntryID) (Payload, error)
	ReadLastEntry() (EntryID, Payload, error)
	GetLastAddConfirmed() EntryID

	GetLedgerID() LedgerID
}

type inMemoryLedger struct {
	id            LedgerID
	bookies       []Bookie
	mu            sync.RWMutex
	lastConfirmed EntryID
}

func (l *inMemoryLedger) GetLedgerID() LedgerID {
	return l.id
}

func (l *inMemoryLedger) AddEntry(data Payload, option LedgerOption) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	var entryIDs []EntryID

	// Ledger ensures data durability and consistency by managing replicas across different bookies.
	// Broker initiates the write operation but doesn't directly handle replication details.
	for i := 0; i < option.WriteQuorum; i++ {
		// 实际上是并发写，each bookie is peer node, coordination is on client/Ledger side
		entryID, err := l.bookies[i].AddEntry(l.id, data)
		if err != nil {
			return 0, err
		}

		entryIDs = append(entryIDs, entryID)
	}

	// Ensure all entryIDs are the same across bookies
	// or handle discrepancies due to failures

	l.lastConfirmed = entryIDs[0]

	return l.lastConfirmed, nil
}

func (l *inMemoryLedger) ReadEntry(entryID EntryID) (Payload, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	return l.bookies[0].ReadEntry(l.id, entryID)
}

func (l *inMemoryLedger) ReadLastEntry() (EntryID, Payload, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.lastConfirmed == -1 {
		return 0, nil, fmt.Errorf("ledger is empty")
	}

	data, err := l.bookies[0].ReadEntry(l.id, l.lastConfirmed)
	if err != nil {
		return 0, nil, err
	}
	return l.lastConfirmed, data, nil
}

func (l *inMemoryLedger) GetLastAddConfirmed() EntryID {
	return l.lastConfirmed
}
