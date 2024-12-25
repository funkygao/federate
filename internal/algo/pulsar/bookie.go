package main

import (
	"fmt"
	"sync"
)

// Bookie 接口定义了 BookKeeper 集群中单个节点的操作
// Bookie 只关心 Ledger 和 Entry 的存储，不关心 Topic/Partition/TimeSegmen
type Bookie interface {
	CreateLedger(ledgerID LedgerID) (Ledger, error)
	DeleteLedger(ledgerID LedgerID) error
	GetLedger(ledgerID LedgerID) (Ledger, error)
}

type BookieError struct {
	Op  string
	Err error
}

func (e *BookieError) Error() string {
	return fmt.Sprintf("bookie %s error: %v", e.Op, e.Err)
}

type InMemoryBookie struct {
	ledgers map[LedgerID]Ledger
	mu      sync.RWMutex
}

func NewInMemoryBookie() *InMemoryBookie {
	return &InMemoryBookie{
		ledgers: make(map[LedgerID]Ledger),
	}
}

func (b *InMemoryBookie) CreateLedger(ledgerID LedgerID) (Ledger, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.ledgers[ledgerID]; exists {
		return nil, fmt.Errorf("ledger already exists")
	}

	ledger := newInMemoryLedger(ledgerID)
	b.ledgers[ledgerID] = ledger
	return ledger, nil
}

func (b *InMemoryBookie) DeleteLedger(ledgerID LedgerID) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.ledgers[ledgerID]; !exists {
		return fmt.Errorf("ledger not found")
	}

	delete(b.ledgers, ledgerID)
	return nil
}

func (b *InMemoryBookie) GetLedger(ledgerID LedgerID) (Ledger, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	ledger, exists := b.ledgers[ledgerID]
	if !exists {
		return nil, fmt.Errorf("ledger not found")
	}

	return ledger, nil
}
