package main

import (
	"sync"
)

type IndexValue struct {
	Offset int64
	Size   int32
}

type IndexCache interface {
	Get(ledgerID LedgerID, entryID EntryID) (IndexValue, bool)
	Put(ledgerID LedgerID, entryID EntryID, entry IndexValue)
}

type indexCache struct {
	cache    map[LedgerID]map[EntryID]IndexValue
	capacity int
	mu       sync.RWMutex
}

func NewIndexCache(capacity int) IndexCache {
	return &indexCache{
		cache:    make(map[LedgerID]map[EntryID]IndexValue),
		capacity: capacity,
	}
}

func (c *indexCache) Get(ledgerID LedgerID, entryID EntryID) (IndexValue, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if ledgerCache, ok := c.cache[ledgerID]; ok {
		if entry, ok := ledgerCache[entryID]; ok {
			return entry, true
		}
	}
	return IndexValue{}, false
}

func (c *indexCache) Put(ledgerID LedgerID, entryID EntryID, entry IndexValue) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, ok := c.cache[ledgerID]; !ok {
		c.cache[ledgerID] = make(map[EntryID]IndexValue)
	}
	c.cache[ledgerID][entryID] = entry

	// 简单的容量控制，实际实现可能需要更复杂的淘汰策略
	if len(c.cache) > c.capacity {
		// 移除最旧的 ledger
		var oldestLedger LedgerID
		for lid := range c.cache {
			oldestLedger = lid
			break
		}
		delete(c.cache, oldestLedger)
	}
}
