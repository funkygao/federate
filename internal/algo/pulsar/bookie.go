package main

// Bookie 是 BookKeeper 集群中的单个节点
type Bookie interface {
	AddEntry(ledgerID LedgerID, entryID EntryID, entry Payload) error
	ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error)
}
