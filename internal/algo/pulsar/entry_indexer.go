package main

import (
	"sync"
)

type IndexValue struct {
	Offset int64
}

type EntryIndexer interface {
	Get(ledgerID LedgerID, entryID EntryID) (IndexValue, bool)
	Put(ledgerID LedgerID, entryID EntryID, entry IndexValue)
}

// 保存(LedgerId, EntryId)到Offset的映射关系，实际上用的是 RocksDB
type entryIndexer struct {
	cache map[LedgerID]map[EntryID]IndexValue
	mu    sync.RWMutex
}

func NewEntryIndexer() EntryIndexer {
	return &entryIndexer{cache: make(map[LedgerID]map[EntryID]IndexValue)}
}

func (c *entryIndexer) Get(ledgerID LedgerID, entryID EntryID) (IndexValue, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if ledgerCache, ok := c.cache[ledgerID]; ok {
		if entry, ok := ledgerCache[entryID]; ok {
			return entry, true
		}
	}
	return IndexValue{}, false
}

func (c *entryIndexer) Put(ledgerID LedgerID, entryID EntryID, entry IndexValue) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, ok := c.cache[ledgerID]; !ok {
		c.cache[ledgerID] = make(map[EntryID]IndexValue)
	}
	c.cache[ledgerID][entryID] = entry
}
