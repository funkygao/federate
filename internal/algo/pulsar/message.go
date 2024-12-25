package main

import "time"

type MessageID struct {
	LedgerID  int64
	EntryID   int64
	Partition int
}

type Message struct {
	ID        MessageID
	Content   string
	Timestamp time.Time
	Delay     time.Duration
}
