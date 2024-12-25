package main

import (
	"sync"
)

// Responsible for bookie load balancing by deciding which bookies store which ledgers.
type BookKeeper interface {
	CreateLedger() (Ledger, error)
	DeleteLedger(ledgerID LedgerID) error
	OpenLedger(ledgerID LedgerID) (Ledger, error)
}

type InMemoryBookKeeper struct {
	bookies []Bookie
	mu      sync.RWMutex

	// BookKeeper 需要在整个集群范围内保证 LedgerID 的唯一性：通过 zk
	nextLedgerID LedgerID
}

func NewInMemoryBookKeeper(numBookies int) *InMemoryBookKeeper {
	bookies := make([]Bookie, numBookies)
	for i := 0; i < numBookies; i++ {
		bookies[i] = NewInMemoryBookie()
	}
	return &InMemoryBookKeeper{
		bookies:      bookies,
		nextLedgerID: 1,
	}
}

func (bk *InMemoryBookKeeper) CreateLedger() (Ledger, error) {
	ledgerID := bk.allocateLedgerID()
	bookie := bk.selectBookie(ledgerID)
	return bookie.CreateLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) DeleteLedger(ledgerID LedgerID) error {
	bookie := bk.selectBookie(ledgerID)
	return bookie.DeleteLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) OpenLedger(ledgerID LedgerID) (Ledger, error) {
	bookie := bk.selectBookie(ledgerID)
	return bookie.GetLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) allocateLedgerID() LedgerID {
	return bk.nextLedgerID.Next()
}

func (bk *InMemoryBookKeeper) selectBookie(ledgerID LedgerID) Bookie {
	return bk.bookies[0]
}
