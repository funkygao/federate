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

	// LedgerID 是由 BookKeeper 服务管理的，而不是单个的 Bookie
	// BookKeeper 需要在整个集群范围内保证 LedgerID 的唯一性
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

func (bk *InMemoryBookKeeper) allocateLedgerID() LedgerID {
	return bk.nextLedgerID.Next()
}

func (bk *InMemoryBookKeeper) CreateLedger() (Ledger, error) {
	// 简化实现：只在第一个 bookie 上创建 ledger
	ledgerID := bk.allocateLedgerID()
	return bk.bookies[0].CreateLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) DeleteLedger(ledgerID LedgerID) error {
	return bk.bookies[0].DeleteLedger(ledgerID)
}

func (bk *InMemoryBookKeeper) OpenLedger(ledgerID LedgerID) (Ledger, error) {
	return bk.bookies[0].GetLedger(ledgerID)
}
