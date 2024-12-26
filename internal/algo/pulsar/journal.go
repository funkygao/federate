package main

import "sync"

type Journal struct {
	entries []JournalEntry
	mu      sync.Mutex
}

type JournalEntry struct {
	ledgerID LedgerID
	entryID  EntryID
	data     Payload
}

func (j *Journal) Append(ledgerID LedgerID, entryID EntryID, data Payload) error {
	j.mu.Lock()
	j.entries = append(j.entries, JournalEntry{
		ledgerID: ledgerID,
		entryID:  entryID,
		data:     data,
	})
	j.mu.Unlock()
	return nil
}
