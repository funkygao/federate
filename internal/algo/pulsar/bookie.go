package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Bookie: Stores entries for ledgers, know nothing about topics: just kv storage.
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
	AddEntry(ledgerID LedgerID, data Payload) (EntryID, error)
	ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error)
}

type bookie struct {
	id int

	memtable    map[LedgerID]map[EntryID]Payload
	nextEntryID map[LedgerID]EntryID
	mu          sync.RWMutex

	journal Journal

	entryLogs      map[LedgerID][]*EntryLog
	activeLogs     map[LedgerID]*EntryLog
	nextEntryLogID EntryLogID
}

func NewBookie(id int) Bookie {
	journal, err := NewFileJournal("journal/")
	if err != nil {
		log.Fatalf("%v", err)
	}

	return &bookie{
		id:             id,
		memtable:       make(map[LedgerID]map[EntryID]Payload),
		entryLogs:      make(map[LedgerID][]*EntryLog),
		activeLogs:     make(map[LedgerID]*EntryLog),
		journal:        journal,
		nextEntryLogID: 0,
		nextEntryID:    make(map[LedgerID]EntryID),
	}
}

func (b *bookie) AddEntry(ledgerID LedgerID, data Payload) (EntryID, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Assign EntryID based on last EntryID
	entryID := b.allocateEntryID(ledgerID)

	// Write to journal first
	if err := b.journal.Append(JournalEntry{ledgerID, entryID, data}); err != nil {
		return entryID, err
	}

	// Then update write cache
	if _, exists := b.memtable[ledgerID]; !exists {
		b.memtable[ledgerID] = make(map[EntryID]Payload)
	}
	b.memtable[ledgerID][entryID] = data

	log.Printf("Bookie[%d] AddEntry(LedgerID = %d, EntryID = %d), data: %s", b.id, ledgerID, entryID, string(data))

	// Async Write EntryLog, ACK Request ASAP
	go b.addEntryLog(ledgerID, entryID, data)

	return entryID, nil
}

func (b *bookie) addEntryLog(ledgerID LedgerID, entryID EntryID, data Payload) error {
	entryLog, err := b.getOrCreateEntryLog(ledgerID)
	if err != nil {
		return err
	}

	log.Printf("Bookie[%d] AddEntry (EntryLog:%d, LedgerID:%d, EntryID:%d): EntryLog written", b.id, entryLog.ID, ledgerID, entryID)

	// 定时刷盘
	entryLog.Write(data)

	// Check if we need to rotate the EntryLog
	if entryLog.Size >= DefaultEntryLogSize || time.Since(entryLog.CreatedAt) >= DefaultRotationInterval {
		if err := b.rotateEntryLog(ledgerID); err != nil {
			return entryID, err
		}
		log.Printf("Bookie[%d] AddEntry (EntryLog:%d, LedgerID:%d, EntryID:%d): EntryLog rotated", b.id, entryLog.ID, ledgerID, entryID)
	}
	return nil
}

func (b *bookie) ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// 先从内存取，没有再 EntryLog
	ledger, exists := b.memtable[ledgerID]
	if !exists {
		return nil, fmt.Errorf("ledger %d not found", ledgerID)
	}

	entry, exists := ledger[entryID]
	if !exists {
		return nil, fmt.Errorf("entry %d not found", entryID)
	}

	// read from entry log

	log.Printf("Bookie[%d] ReadEntry(LedgerID:%d, EntryID:%d): entry loaded", b.id, ledgerID, entryID)
	return entry, nil
}

func (b *bookie) allocateEntryID(ledgerID LedgerID) EntryID {
	if _, ok := b.nextEntryID[ledgerID]; !ok {
		b.nextEntryID[ledgerID] = 0
	}

	return b.nextEntryID[ledgerID].Next()
}

func (b *bookie) getOrCreateEntryLog(ledgerID LedgerID) (*EntryLog, error) {
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

func (b *bookie) rotateEntryLog(ledgerID LedgerID) error {
	newLog, err := b.getOrCreateEntryLog(ledgerID)
	if err != nil {
		return err
	}

	b.activeLogs[ledgerID] = newLog
	return nil
}
