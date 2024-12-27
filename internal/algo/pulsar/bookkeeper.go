package main

import (
	"fmt"
	"sync"
)

// BookKeeper: Manages ledgers and coordinates with bookies for storage.
// Its storage RPC Client for Broker.
type BookKeeper interface {
	CreateLedger(LedgerOption) (Ledger, error)
	OpenLedger(ledgerID LedgerID) (Ledger, error)

	LedgerOption(LedgerID) LedgerOption
}

type bookKeeper struct {
	bookies []Bookie

	ledgers       map[LedgerID]Ledger
	ledgerOptions map[LedgerID]LedgerOption
	mu            sync.RWMutex

	// BookKeeper 需要在整个集群范围内保证 LedgerID 的唯一性：通过 zk
	nextLedgerID LedgerID

	zk ZooKeeper
}

func NewBookKeeper(clusterSize int) *bookKeeper {
	bookies := make([]Bookie, clusterSize)
	for i := 0; i < clusterSize; i++ {
		bookies[i] = NewBookie(i)
	}
	return &bookKeeper{
		bookies:       bookies,
		ledgers:       make(map[LedgerID]Ledger),
		ledgerOptions: make(map[LedgerID]LedgerOption),
		nextLedgerID:  0,
		zk:            getZooKeeper(),
	}
}

func (bk *bookKeeper) CreateLedger(opt LedgerOption) (l Ledger, err error) {
	ledgerID := bk.allocateLedgerID()
	l = &ledger{
		id:            ledgerID,
		bookies:       bk.selectBookies(ledgerID, opt),
		lastConfirmed: -1,
	}
	bk.ledgers[ledgerID] = l
	bk.ledgerOptions[ledgerID] = opt

	return
}

func (bk *bookKeeper) OpenLedger(ledgerID LedgerID) (Ledger, error) {
	bk.mu.RLock()
	defer bk.mu.RUnlock()

	ledger, exists := bk.ledgers[ledgerID]
	if !exists {
		return nil, fmt.Errorf("ledger %d not found", ledgerID)
	}
	return ledger, nil
}

func (bk *bookKeeper) LedgerOption(ledgerID LedgerID) LedgerOption {
	return bk.ledgerOptions[ledgerID]
}

func (bk *bookKeeper) allocateLedgerID() LedgerID {
	if prodEnv {
		return bk.zk.NextLedgerID()
	}

	return bk.nextLedgerID.Next()
}

func (bk *bookKeeper) selectBookies(ledgerID LedgerID, opt LedgerOption) []Bookie {
	// 实际会考虑负载等策略进行选择
	selected := make([]Bookie, opt.EnsembleSize)
	for i := 0; i < opt.EnsembleSize; i++ {
		selected[i] = bk.bookies[i%len(bk.bookies)]
	}

	// 选择后不能变了，除非节点崩溃
	bk.zk.RegisterLedger(LedgerInfo{ledgerID, opt, selected})

	return selected
}
