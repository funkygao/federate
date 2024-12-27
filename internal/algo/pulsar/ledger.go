package main

import (
	"fmt"
	"log"
	"sync"
)

type LedgerOption struct {
	EnsembleSize int
	WriteQuorum  int
	AckQuorum    int
}

// Ledger: Represents a sequence of entries.
type Ledger interface {
	AddEntry(Payload, LedgerOption) (EntryID, error)
	ReadEntry(EntryID) (Payload, error)
	ReadLastEntry() (EntryID, Payload, error)
	GetLastAddConfirmed() EntryID

	// Close后就不能写入了，这对 rebalance/failover 重要
	Close()

	GetLedgerID() LedgerID
}

type ledger struct {
	id      LedgerID
	bookies []Bookie
	mu      sync.RWMutex

	// EntryID 是在 Ledger 级别维护和生成的，而不是由单个 Bookie 生成
	// 集中式管理：让 Ledger 管理 EntryID 可以确保在分布式系统中的一致性
	// 避免冲突：如果让每个 Bookie 独立生成 EntryID，可能会导致冲突和不一致
	// 简化恢复过程：easier to determine the last confirmed entry and continue from there
	lastConfirmed EntryID
}

func (l *ledger) GetLedgerID() LedgerID {
	return l.id
}

func (l *ledger) Close() {
}

func (l *ledger) AddEntry(data Payload, option LedgerOption) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// ledger 维护 ledger 范围内的 entryID
	entryID := l.allocateEntryID()

	// Write to bookies concurrently
	results := make(chan error, option.WriteQuorum)
	for i := 0; i < option.WriteQuorum; i++ {
		go func(b Bookie) {
			log.Printf("Ledger[%d] AddEntry to bookie[%d] with WriteQuorum: %d", l.id, i, option.WriteQuorum)
			results <- b.AddEntry(l.id, entryID, data)
		}(l.bookies[i])
	}

	// Wait for results and count successes/failures
	successCount, errorCount := 0, 0
	var lastError error
	for i := 0; i < option.WriteQuorum; i++ {
		if lastError = <-results; lastError == nil {
			successCount++
		} else {
			errorCount++
		}

		if successCount >= option.AckQuorum {
			// bingo!
			l.lastConfirmed = entryID
			return entryID, nil
		}

		// Check if it's impossible to meet AckQuorum: fail fast
		if errorCount > option.WriteQuorum-option.AckQuorum {
			return 0, fmt.Errorf("failed to meet AckQuorum: %v", lastError)
		}
	}

	return 0, lastError
}

func (l *ledger) ReadEntry(entryID EntryID) (Payload, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	log.Printf("Ledger[%d] ReadEntry(%d)", l.id, entryID)

	// 实际上是并发读，取最快的
	return l.bookies[0].ReadEntry(l.id, entryID)
}

func (l *ledger) ReadLastEntry() (EntryID, Payload, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.lastConfirmed == -1 {
		return 0, nil, fmt.Errorf("ledger is empty")
	}

	data, err := l.bookies[0].ReadEntry(l.id, l.lastConfirmed)
	if err != nil {
		return 0, nil, err
	}
	return l.lastConfirmed, data, nil
}

func (l *ledger) GetLastAddConfirmed() EntryID {
	return l.lastConfirmed
}

func (l *ledger) allocateEntryID() EntryID {
	return l.lastConfirmed + 1
}
