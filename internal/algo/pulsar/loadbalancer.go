package main

var (
	_ PartitionLB = (*InMemoryBroker)(nil)
	_ BookieLB    = (*InMemoryBookKeeper)(nil)
)

type PartitionLB interface {
	selectPartition(msg Message) PartitionID
}

type BookieLB interface {
	selectBookie(ledgerID LedgerID) Bookie
}
