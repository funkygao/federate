package main

// 整个分布式存储服务，由多个 Bookie 节点构成
type BookKeeperService interface {
	AddEntry(partitionID PartitionID, timeSegmentID TimeSegmentID, ledgerID LedgerID, entry Payload) (EntryID, error)
	ReadEntry(partitionID PartitionID, timeSegmentID TimeSegmentID, ledgerID LedgerID, entryID EntryID) (Payload, error)

	CreateLedger(partitionID PartitionID, timeSegmentID TimeSegmentID) (LedgerID, error)
}
