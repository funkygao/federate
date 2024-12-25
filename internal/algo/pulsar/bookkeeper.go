package main

import "fmt"

// BookKeeper 是整个分布式存储服务，由多个 Bookie 节点构成
type BookKeeper interface {
	CreateLedger() (LedgerID, error)
	DeleteLedger(ledgerID LedgerID) error
	OpenLedger(ledgerID LedgerID) (Ledger, error)
}

type BookKeeperError struct {
	Op  string
	Err error
}

func (e *BookKeeperError) Error() string {
	return fmt.Sprintf("bookkeeper %s error: %v", e.Op, e.Err)
}

// TODO not used
type ManagedLedger interface {
	AddEntry(payload []byte) (PositionImpl, error)
	ReadEntries(startPosition PositionImpl, numberOfEntries int64) ([]Entry, error)
	DeleteLedger(ledgerID int64) error
}

// TODO not used
type PositionImpl struct {
	LedgerID int64
	EntryID  int64
}

// TODO not used
type Entry struct {
	Position PositionImpl
	Payload  []byte
}
