package main

import (
	"fmt"
)

// Bookie 是 BookKeeper 集群中的单个节点
type Bookie interface {
	CreateLedger() (LedgerID, error)
	DeleteLedger(ledgerID LedgerID) error

	AddEntry(ledgerID LedgerID, entry Payload) (EntryID, error)
	ReadEntry(ledgerID LedgerID, entryID EntryID) (Payload, error)

	GetLastEntryID(ledgerID LedgerID) (EntryID, error)
}

type BookieError struct {
	Op  string
	Err error
}

func (e *BookieError) Error() string {
	return fmt.Sprintf("bookie %s error: %v", e.Op, e.Err)
}
