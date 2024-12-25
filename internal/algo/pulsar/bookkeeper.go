package main

import (
	"sync"
)

// BookKeeper 是整个分布式存储服务，由多个 Bookie 节点构成
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
	return bk.lbBookie().CreateLedger(bk.allocateLedgerID())
}

func (bk *InMemoryBookKeeper) DeleteLedger(ledgerID LedgerID) error {
	return bk.locateBookie(ledgerID).DeleteLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) OpenLedger(ledgerID LedgerID) (Ledger, error) {
	return bk.locateBookie(ledgerID).GetLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) allocateLedgerID() LedgerID {
	return bk.nextLedgerID.Next()
}

func (bk *InMemoryBookKeeper) lbBookie() Bookie {
	// 简化实现
	return bk.bookies[0]
}

func (bk *InMemoryBookKeeper) locateBookie(ledgerID LedgerID) Bookie {
	return bk.bookies[0]
}
