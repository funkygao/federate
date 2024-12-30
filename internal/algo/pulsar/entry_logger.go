package main

import (
	"fmt"
	"time"
)

const (
	DefaultEntryLogSize     = 1 << 30 // 1GB
	DefaultRotationInterval = 24 * time.Hour
)

type LedgerEntry struct {
	EntryID   EntryID
	Content   Payload
	CreatedAt time.Time
}

// Ledger disk file 相当于 Kafka Segment，但它增加了 LedgerID，没有绑定死 Broker，就避免了搬迁数据.
type EntryLogger interface {
	Write(LedgerID, LedgerEntry) error
	Read(LedgerID, EntryID) (Payload, error)
}

type entryLogger struct {
	size         int64
	entryIndexer EntryIndexer
}

func NewEntryLogger() EntryLogger {
	return &entryLogger{0, NewEntryIndexer()}
}

func (l *entryLogger) Write(ledgerID LedgerID, log LedgerEntry) error {
	l.size += log.Content.Size()

	// Write disk

	// Write index
	l.entryIndexer.Put(ledgerID, log.EntryID, IndexValue{Offset: l.size})

	// Check if we need to rotate the EntryLog
	if l.size >= DefaultEntryLogSize || time.Since(log.CreatedAt) >= DefaultRotationInterval {
		if err := l.rotate(); err != nil {
			return err
		}
	}

	if prodEnv {
		if err := l.sync(); err != nil {
			return err
		}
	}

	return nil
}

func (l *entryLogger) Read(ledgerID LedgerID, entryID EntryID) (Payload, error) {
	return nil, fmt.Errorf("Read operation not implemented")
}

func (l *entryLogger) rotate() error {
	return nil
}

func (l *entryLogger) sync() error {
	return nil
}
