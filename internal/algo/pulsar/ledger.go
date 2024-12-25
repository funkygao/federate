package main

import (
	"fmt"
	"sync"
)

// Ledger 表示一个连续的日志条目序列
type Ledger interface {
	AddEntry(entry Payload) (EntryID, error)
	ReadEntry(entryID EntryID) (Payload, error)

	GetLastEntryID() (EntryID, error)
}

type inMemoryLedger struct {
	id          LedgerID

	entries     map[EntryID]Payload
	mu          sync.RWMutex

	lastEntryID EntryID
}

func newInMemoryLedger(id LedgerID) *inMemoryLedger {
	return &inMemoryLedger{
		id:          id,
		entries:     make(map[EntryID]Payload),
		lastEntryID: 0,
	}
}

func (l *inMemoryLedger) AddEntry(entry Payload) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.lastEntryID++
	entryID := l.lastEntryID
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

func (l *inMemoryLedger) GetLastEntryID() (EntryID, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	return l.lastEntryID, nil
}
