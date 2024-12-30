package main

var (
	_ PartitionLB       = (*broker)(nil)
	_ BookieLB          = (*bookKeeper)(nil)
	_ LedgerIDAllocator = (*bookKeeper)(nil)
	_ EntryIDAllocator  = (*ledger)(nil)
)

type PartitionLB interface {
	selectPartition(*Topic, Message) PartitionID
}

type LedgerIDAllocator interface {
	allocateLedgerID() LedgerID
}

type EntryIDAllocator interface {
	allocateEntryID() EntryID
}

type BookieLB interface {
	selectBookies(LedgerID, LedgerOption) []Bookie
}
