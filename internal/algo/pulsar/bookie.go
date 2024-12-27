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
	ID() int

	AddEntry(ledgerID LedgerID, entryID EntryID, data Payload) error
	ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error)

	DeleteLedgerData(LedgerID) error
}

type bookie struct {
	id int

	memtable    map[LedgerID]map[EntryID]Payload
	nextEntryID map[LedgerID]*EntryID
	mu          sync.RWMutex

	journal    Journal
	indexCache IndexCache

	entryLogs      map[LedgerID][]*EntryLog
	activeLogs     map[LedgerID]*EntryLog
	nextEntryLogID EntryLogID
}

func NewBookie(id int) Bookie {
	b := &bookie{
		id:             id,
		memtable:       make(map[LedgerID]map[EntryID]Payload),
		entryLogs:      make(map[LedgerID][]*EntryLog),
		activeLogs:     make(map[LedgerID]*EntryLog),
		nextEntryLogID: 0,
		indexCache:     NewIndexCache(10000),
		nextEntryID:    make(map[LedgerID]*EntryID),
	}

	journal, err := NewFileJournal("journal/", b)
	if err != nil {
		log.Fatalf("%v", err)
	}
	b.journal = journal

	return b
}

func (b *bookie) ID() int {
	return b.id
}

func (b *bookie) AddEntry(ledgerID LedgerID, entryID EntryID, data Payload) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Write to journal first
	if err := b.journal.Append(JournalEntry{ledgerID, entryID, data}); err != nil {
		return err
	}

	// Then update write cache
	if _, exists := b.memtable[ledgerID]; !exists {
		b.memtable[ledgerID] = make(map[EntryID]Payload)
	}
	b.memtable[ledgerID][entryID] = data

	var offset int64
	b.indexCache.Put(ledgerID, entryID, IndexValue{Offset: offset, Size: int32(len(data))})

	log.Printf("Bookie[%d] AddEntry(LedgerID = %d, EntryID = %d), data: %s", b.id, ledgerID, entryID, string(data))

	// Async Write EntryLog, ACK Request ASAP
	go b.addEntryLog(ledgerID, entryID, data)

	return nil
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
			return err
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

func (b *bookie) DeleteLedgerData(ledgerID LedgerID) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.memtable[ledgerID]; exists {
		delete(b.memtable, ledgerID)
		log.Printf("Bookie[%d] Deleted data for ledger %d", b.id, ledgerID)
	}

	// Delete entry logs and other files associated with the ledger
	// Implement file deletion logic here if you have persisted data to disk

	return nil
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
