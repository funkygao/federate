package main

import (
	"fmt"
	"sync"
)

type InMemoryBookKeeper struct {
	ledgers map[LedgerID]*inMemoryLedger
	mu      sync.RWMutex
}

func NewInMemoryBookKeeper() *InMemoryBookKeeper {
	return &InMemoryBookKeeper{
		ledgers: make(map[LedgerID]*inMemoryLedger),
	}
}

func (bk *InMemoryBookKeeper) CreateLedger() (LedgerID, error) {
	bk.mu.Lock()
	defer bk.mu.Unlock()

	ledgerID := LedgerID(len(bk.ledgers) + 1) // LedgerID 如何生成
	bk.ledgers[ledgerID] = newInMemoryLedger(ledgerID)
	return ledgerID, nil
}

func (bk *InMemoryBookKeeper) DeleteLedger(ledgerID LedgerID) error {
	bk.mu.Lock()
	defer bk.mu.Unlock()

	if _, exists := bk.ledgers[ledgerID]; !exists {
		return fmt.Errorf("ledger not found")
	}
	delete(bk.ledgers, ledgerID)
	return nil
}

func (bk *InMemoryBookKeeper) OpenLedger(ledgerID LedgerID) (Ledger, error) {
	bk.mu.RLock()
	defer bk.mu.RUnlock()

	ledger, exists := bk.ledgers[ledgerID]
	if !exists {
		return nil, fmt.Errorf("ledger not found")
	}
	return ledger, nil
}
