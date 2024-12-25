package main

import (
	"fmt"
	"sync"
)

type InMemoryBookie struct {
	storage map[LedgerID]map[EntryID]Payload
	mu      sync.RWMutex
}

func NewInMemoryBookie() *InMemoryBookie {
	return &InMemoryBookie{
		storage: make(map[LedgerID]map[EntryID]Payload),
	}
}

func (b *InMemoryBookie) AddEntry(ledgerID LedgerID, entryID EntryID, entry Payload) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, ok := b.storage[ledgerID]; !ok {
		b.storage[ledgerID] = make(map[EntryID]Payload)
	}

	b.storage[ledgerID][entryID] = entry
	return nil
}

func (b *InMemoryBookie) ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if ledger, ok := b.storage[ledgerID]; ok {
		if entry, ok := ledger[entryID]; ok {
			return entry, nil
		}
	}
	return nil, fmt.Errorf("entry not found")
}
