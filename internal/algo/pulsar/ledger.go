package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

type LedgerOption struct {
	EnsembleSize int
	WriteQuorum  int
	AckQuorum    int
}

// Ledgers serve a similar role to segments in Kafka: but not bound to physical Parition.
// They are units of storage that can be individually managed, rolled over, and deleted.
// Each ledger contains a sequence of entries (messages) and is written to a set of Bookies.
type Ledger interface {
	AddEntry(Payload, LedgerOption) (EntryID, error)
	ReadEntry(EntryID) (Payload, error)
	ReadLastEntry() (EntryID, Payload, error)
	GetLastAddConfirmed() EntryID

	// Close后就不能写入了，这对 rebalance/failover 重要
	Close()

	Age() time.Duration
	Size() int64

	GetLedgerID() LedgerID
	Bookies() []Bookie
}

type ledger struct {
	id            LedgerID
	bookies       []Bookie
	mu            sync.RWMutex
	lastConfirmed EntryID

	createdAt time.Time
	size      int64
}

func (l *ledger) GetLedgerID() LedgerID {
	return l.id
}

func (l *ledger) Close() {
	log.Printf("Ledger[%d] is closed", l.id)
}

func (l *ledger) Bookies() []Bookie {
	return l.bookies
}

func (l *ledger) Age() time.Duration {
	return time.Since(l.createdAt)
}

func (l *ledger) Size() int64 {
	return l.size
}

func (l *ledger) AddEntry(data Payload, option LedgerOption) (EntryID, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// EntryID 是在 Ledger 级别维护和生成的，而不是由单个 Bookie 生成
	// 集中式管理：让 Ledger 管理 EntryID 可以确保在分布式系统中的一致性
	// 避免冲突：如果让每个 Bookie 独立生成 EntryID，可能会导致冲突和不一致
	// 简化恢复过程：easier to determine the last confirmed entry and continue from there
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
			l.size += data.Size()
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

	type readReult struct {
		payload Payload
		err     error
	}

	// Scatter
	resultChan := make(chan readReult, len(l.bookies))
	for i, bookie := range l.bookies {
		go func(b Bookie, bookieIndex int) {
			payload, err := b.ReadEntry(l.id, entryID)
			resultChan <- readReult{payload, err}
			log.Printf("Ledger[%d] ReadEntry(%d) from Bookie[%d] completed", l.id, entryID, bookieIndex)
		}(bookie, i)
	}

	// Gather
	var lastError error
	for i := 0; i < len(l.bookies); i++ {
		result := <-resultChan
		if result.err == nil {
			// 成功读取，立即返回结果
			return result.payload, nil
		}
		lastError = result.err
	}

	return nil, fmt.Errorf("failed to read entry %d from all bookies: %v", entryID, lastError)
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
