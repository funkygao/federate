package main

import (
	"time"
)

const (
	DefaultEntryLogSize     = 1 << 30 // 1GB
	DefaultRotationInterval = 24 * time.Hour
)

type EntryLog struct {
	EntryID   EntryID
	Content   Payload
	CreatedAt time.Time
}

type EntryLogger interface {
	Write(LedgerID, EntryLog) error
}

type entryLogger struct {
	size       int64
	entryIndex EntryIndex
}

func NewEntryLogger() EntryLogger {
	return &entryLogger{0, NewEntryIndex()}
}

func (l *entryLogger) Write(ledgerID LedgerID, log EntryLog) error {
	l.size += log.Content.Size()

	// Write disk

	// Write index
	l.entryIndex.Put(ledgerID, log.EntryID, IndexValue{Offset: l.size})

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

func (l *entryLogger) rotate() error {
	return nil
}

func (l *entryLogger) sync() error {
	return nil
}
