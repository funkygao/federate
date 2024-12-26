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

// Ledger 表示一个连续的日志条目序列
type Ledger interface {
	AddEntry(Payload) (EntryID, error)
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

	bookKeeper *InMemoryBookKeeper
}

func (l *inMemoryLedger) GetLedgerID() LedgerID {
	return l.id
}

func (l *inMemoryLedger) AddEntry(data Payload) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entryID := l.allocateEntryID()

	options := l.bookKeeper.ledgerOptions[l.id]
	for i := 0; i < options.WriteQuorum; i++ {
		err := l.bookies[i].AddEntry(l.id, entryID, data)
		if err != nil {
			return 0, err
		}
	}

	l.lastConfirmed = entryID

	return entryID, nil
}

func (l *inMemoryLedger) allocateEntryID() EntryID {
	return l.lastConfirmed.Next()
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
