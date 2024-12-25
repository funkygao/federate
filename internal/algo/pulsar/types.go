package main

import "time"

type LedgerID int64
type EntryID int64
type PartitionID int
type TimeSegmentID int64

type Payload []byte

type MessageID struct {
	LedgerID      LedgerID
	EntryID       EntryID
	PartitionID   PartitionID
	TimeSegmentID TimeSegmentID
}

type Message struct {
	ID        MessageID
	Content   Payload
	Timestamp time.Time
	Delay     time.Duration
}

type Ledger struct {
	ID      LedgerID
	Entries map[EntryID]Payload
}

type TimeSegment struct {
	ID      TimeSegmentID
	Ledgers []Ledger
}

type Partition struct {
	ID           PartitionID
	TimeSegments []TimeSegment
}
