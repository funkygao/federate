package main

import (
	"fmt"
	"sync"
)

type InMemoryBookie struct {
	ledgers map[LedgerID]*inMemoryLedger
	mu      sync.RWMutex

	nextLedgerID LedgerID
}

// TODO not used
func NewInMemoryBookie() *InMemoryBookie {
	return &InMemoryBookie{
		ledgers:      make(map[LedgerID]*inMemoryLedger),
		nextLedgerID: 1,
	}
}

func (b *InMemoryBookie) CreateLedger() (LedgerID, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	ledgerID := b.nextLedgerID
	b.nextLedgerID++

	b.ledgers[ledgerID] = &inMemoryLedger{
		id:      ledgerID,
		entries: make(map[EntryID]Payload),
	}

	return ledgerID, nil
}

func (b *InMemoryBookie) DeleteLedger(ledgerID LedgerID) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.ledgers[ledgerID]; !exists {
		return &BookieError{Op: "DeleteLedger", Err: fmt.Errorf("ledger not found")}
	}

	delete(b.ledgers, ledgerID)
	return nil
}

func (b *InMemoryBookie) AddEntry(ledgerID LedgerID, entry Payload) (EntryID, error) {
	b.mu.RLock()
	ledger, exists := b.ledgers[ledgerID]
	b.mu.RUnlock()

	if !exists {
		return 0, &BookieError{Op: "AddEntry", Err: fmt.Errorf("ledger not found")}
	}

	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	entryID := ledger.lastEntryID + 1
	ledger.entries[entryID] = entry
	ledger.lastEntryID = entryID

	return entryID, nil
}

func (b *InMemoryBookie) ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error) {
	b.mu.RLock()
	ledger, exists := b.ledgers[ledgerID]
	b.mu.RUnlock()

	if !exists {
		return nil, &BookieError{Op: "ReadEntry", Err: fmt.Errorf("ledger not found")}
	}

	ledger.mu.RLock()
	defer ledger.mu.RUnlock()

	entry, exists := ledger.entries[entryID]
	if !exists {
		return nil, &BookieError{Op: "ReadEntry", Err: fmt.Errorf("entry not found")}
	}

	return entry, nil
}

func (b *InMemoryBookie) GetLastEntryID(ledgerID LedgerID) (EntryID, error) {
	b.mu.RLock()
	ledger, exists := b.ledgers[ledgerID]
	b.mu.RUnlock()

	if !exists {
		return 0, &BookieError{Op: "GetLastEntryID", Err: fmt.Errorf("ledger not found")}
	}

	ledger.mu.RLock()
	defer ledger.mu.RUnlock()

	return ledger.lastEntryID, nil
}
