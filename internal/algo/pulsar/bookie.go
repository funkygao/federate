package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Bookie: Stores entries for ledgers, knows nothing about topics: just kv storage.
type Bookie interface {
	ID() int

	AddEntry(LedgerID, EntryID, Payload) error
	ReadEntry(LedgerID, EntryID) (Payload, error)

	DeleteLedgerData(LedgerID) error
}

type bookie struct {
	id int

	nextEntryID map[LedgerID]*EntryID
	mu          sync.RWMutex

	journal     EntryJournal
	memtable    map[LedgerID]map[EntryID]Payload // for trailing reads
	entryLogger EntryLogger
}

func NewBookie(id int) Bookie {
	b := &bookie{
		id:          id,
		memtable:    make(map[LedgerID]map[EntryID]Payload),
		entryLogger: NewEntryLogger(),
		nextEntryID: make(map[LedgerID]*EntryID),
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

	// Write to journal first: WAL
	if err := b.journal.Append(JournalEntry{ledgerID, entryID, data}); err != nil {
		return err
	}

	// Then update write cache
	if _, exists := b.memtable[ledgerID]; !exists {
		b.memtable[ledgerID] = make(map[EntryID]Payload)
	}
	b.memtable[ledgerID][entryID] = data

	log.Printf("Bookie[%d] AddEntry(LedgerID = %d, EntryID = %d), data: %s", b.id, ledgerID, entryID, string(data))

	// Async Write LedgerEntry, ACK Request ASAP
	go b.addLedgerEntry(ledgerID, entryID, data)

	return nil
}

func (b *bookie) addLedgerEntry(ledgerID LedgerID, entryID EntryID, data Payload) error {
	ledgerEntry := LedgerEntry{
		EntryID:   entryID,
		Content:   data,
		CreatedAt: time.Now(),
	}

	log.Printf("Bookie[%d] AddEntry (LedgerID:%d, EntryID:%d): LedgerEntry written", b.id, ledgerID, entryID)

	// 定时刷盘
	b.entryLogger.Write(ledgerID, ledgerEntry)

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

	// for test only
	entry, exists := ledger[entryID]
	if !exists {
		return nil, fmt.Errorf("entry %d not found", entryID)
	}

	// read from entry log
	b.entryLogger.Read(ledgerID, entryID)

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
