package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Bookie interface defines the operations that a bookie should support.
// The physical storage layout of a Bookie on a file system typically looks like this:
//
//	/bookie-data/
//	│
//	├── journal/
//	│   ├── current/
//	│   │   └── ${journal-id}  (mmap)
//	│   └── ${journal-id}.txn
//	│
//	├── ledgers/
//	│   ├── ${ledger-id-group1}/
//	│   │   ├── ${ledger-id1}/
//	│   │   │   ├── entry-log-${entrylog-id}  (Similar to Kafka's segment)
//	│   │   │   └── index
//	│   │   └── ${ledger-id2}/
//	│   │       ├── entry-log-${entrylog-id}
//	│   │       └── index
//	│   │
//	│   └── ${ledger-id-group2}/
//	│       └── ...
//	│
//	└── meta/
//	    └── ${ledger-id}.meta  (Metadata for each ledger)
//
// Note:
// - Entries are grouped into entry logs for efficiency.
// - Ledgers are grouped into subdirectories to avoid too many files in a single directory.
type Bookie interface {
	AddEntry(ledgerID LedgerID, entryID EntryID, data Payload) error
	ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error)
}

type InMemoryBookie struct {
	entries map[LedgerID]map[EntryID]Payload
	mu      sync.RWMutex

	entryLogs      map[LedgerID][]*EntryLog
	activeLogs     map[LedgerID]*EntryLog
	nextEntryLogID EntryLogID
}

func NewInMemoryBookie() Bookie {
	return &InMemoryBookie{
		entries:        make(map[LedgerID]map[EntryID]Payload),
		entryLogs:      make(map[LedgerID][]*EntryLog),
		activeLogs:     make(map[LedgerID]*EntryLog),
		nextEntryLogID: 0,
	}
}

func (b *InMemoryBookie) AddEntry(ledgerID LedgerID, entryID EntryID, data Payload) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.entries[ledgerID]; !exists {
		b.entries[ledgerID] = make(map[EntryID]Payload)
	}
	b.entries[ledgerID][entryID] = data

	// EntryLog
	entryLog, err := b.getOrCreateEntryLog(ledgerID)
	if err != nil {
		return err
	}

	log.Printf("AddEntry/%d [L:%d,E:%d]", entryLog.ID, ledgerID, entryID)
	entryLog.Write(data)

	// Check if we need to rotate the EntryLog
	if entryLog.Size >= DefaultEntryLogSize || time.Since(entryLog.CreatedAt) >= DefaultRotationInterval {
		if err := b.rotateEntryLog(ledgerID); err != nil {
			return err
		}
	}

	return nil
}

func (b *InMemoryBookie) ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	ledger, exists := b.entries[ledgerID]
	if !exists {
		return nil, fmt.Errorf("ledger not found")
	}

	entry, exists := ledger[entryID]
	if !exists {
		return nil, fmt.Errorf("entry not found")
	}

	return entry, nil
}

func (b *InMemoryBookie) getOrCreateEntryLog(ledgerID LedgerID) (*EntryLog, error) {
	if currentLog, exists := b.activeLogs[ledgerID]; exists {
		return currentLog, nil
	}

	newLog := &EntryLog{
		ID:        b.nextEntryLogID,
		CreatedAt: time.Now(),
	}
	b.nextEntryLogID.Next()

	b.entryLogs[ledgerID] = append(b.entryLogs[ledgerID], newLog)
	b.activeLogs[ledgerID] = newLog

	return newLog, nil
}

func (b *InMemoryBookie) rotateEntryLog(ledgerID LedgerID) error {
	newLog, err := b.getOrCreateEntryLog(ledgerID)
	if err != nil {
		return err
	}

	b.activeLogs[ledgerID] = newLog
	return nil
}
