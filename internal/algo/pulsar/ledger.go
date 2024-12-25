package main

import (
	"fmt"
	"sync"
)

// Ledger 表示一个连续的日志条目序列
type Ledger interface {
	AddEntry(entry Payload) (EntryID, error)
	ReadEntry(entryID EntryID) (Payload, error)

	GetID() LedgerID
}

type inMemoryLedger struct {
	id LedgerID

	entries map[EntryID]Payload
	mu      sync.RWMutex

	lastEntryID EntryID
}

func newInMemoryLedger(id LedgerID) *inMemoryLedger {
	return &inMemoryLedger{
		id:      id,
		entries: make(map[EntryID]Payload),
	}
}

func (l *inMemoryLedger) GetID() LedgerID {
	return l.id
}

func (l *inMemoryLedger) AddEntry(entry Payload) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entryID := l.lastEntryID.Next()
	l.entries[entryID] = entry
	return entryID, nil
}

func (l *inMemoryLedger) ReadEntry(entryID EntryID) (Payload, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	entry, exists := l.entries[entryID]
	if !exists {
		return nil, fmt.Errorf("entry not found")
	}

	return entry, nil
}
