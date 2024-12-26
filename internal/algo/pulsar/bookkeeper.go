package main

import (
	"fmt"
	"sync"
)

// Responsible for bookie load balancing by deciding which bookies store which ledgers.
type BookKeeper interface {
	CreateLedger(LedgerOption) (Ledger, error)
	DeleteLedger(ledgerID LedgerID) error
	OpenLedger(ledgerID LedgerID) (Ledger, error)
}

type InMemoryBookKeeper struct {
	bookies       []Bookie
	ledgers       map[LedgerID]Ledger
	ledgerOptions map[LedgerID]LedgerOption
	mu            sync.RWMutex

	// BookKeeper 需要在整个集群范围内保证 LedgerID 的唯一性：通过 zk
	nextLedgerID LedgerID
}

func NewInMemoryBookKeeper(clusterSize int) *InMemoryBookKeeper {
	bookies := make([]Bookie, clusterSize)
	for i := 0; i < clusterSize; i++ {
		bookies[i] = NewInMemoryBookie()
	}
	return &InMemoryBookKeeper{
		bookies:       bookies,
		ledgers:       make(map[LedgerID]Ledger),
		ledgerOptions: make(map[LedgerID]LedgerOption),
		nextLedgerID:  1,
	}
}

func (bk *InMemoryBookKeeper) CreateLedger(opt LedgerOption) (ledger Ledger, err error) {
	ledgerID := bk.allocateLedgerID()
	ledger = &inMemoryLedger{
		id:            ledgerID,
		bookies:       bk.selectBookies(opt.EnsembleSize),
		lastConfirmed: -1,
		bookKeeper:    bk,
	}
	bk.ledgers[ledgerID] = ledger
	bk.ledgerOptions[ledgerID] = opt
	return
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

func (bk *InMemoryBookKeeper) DeleteLedger(ledgerID LedgerID) error {
	bk.mu.Lock()
	defer bk.mu.Unlock()

	if _, exists := bk.ledgers[ledgerID]; !exists {
		return fmt.Errorf("ledger not found")
	}

	delete(bk.ledgers, ledgerID)
	delete(bk.ledgerOptions, ledgerID)
	return nil
}

func (bk *InMemoryBookKeeper) allocateLedgerID() LedgerID {
	return bk.nextLedgerID.Next()
}

func (bk *InMemoryBookKeeper) selectBookies(ensembleSize int) []Bookie {
	selected := make([]Bookie, ensembleSize)
	for i := 0; i < ensembleSize; i++ {
		selected[i] = bk.bookies[i%len(bk.bookies)]
	}
	return selected
}
