package main

type BookKeeper interface {
	AddEntry(ledgerID int64, entry []byte) (int64, error)
	ReadEntry(ledgerID int64, entryID int64) ([]byte, error)
}

type InMemoryBookKeeper struct {
	entries map[int64]map[int64][]byte
}

func NewInMemoryBookKeeper() *InMemoryBookKeeper {
	return &InMemoryBookKeeper{
		entries: make(map[int64]map[int64][]byte),
	}
}

func (bk *InMemoryBookKeeper) AddEntry(ledgerID int64, entry []byte) (int64, error) {
	if _, ok := bk.entries[ledgerID]; !ok {
		bk.entries[ledgerID] = make(map[int64][]byte)
	}
	entryID := int64(len(bk.entries[ledgerID]))
	bk.entries[ledgerID][entryID] = entry
	return entryID, nil
}

func (bk *InMemoryBookKeeper) ReadEntry(ledgerID int64, entryID int64) ([]byte, error) {
	if ledger, ok := bk.entries[ledgerID]; ok {
		if entry, ok := ledger[entryID]; ok {
			return entry, nil
		}
	}
	return nil, nil
}
