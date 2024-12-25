package main

var (
	_ PartitionLB = (*InMemoryBroker)(nil)
	_ BookieLB    = (*InMemoryBookKeeper)(nil)
	// TODO any more load balancer in pulsar?
)

type PartitionLB interface {
	selectPartition(msg Message) PartitionID
}

type BookieLB interface {
	selectBookie(ledgerID LedgerID) Bookie
}
