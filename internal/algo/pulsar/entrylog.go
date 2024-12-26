package main

import (
	"sync/atomic"
	"time"
)

const (
	DefaultEntryLogSize     = 1 << 30 // 1GB
	DefaultRotationInterval = 24 * time.Hour
)

type EntryLogID int64

func (id *EntryLogID) Next() EntryLogID {
	return EntryLogID(atomic.AddInt64((*int64)(id), 1))
}

type EntryLog struct {
	ID        EntryLogID
	Size      int64
	CreatedAt time.Time
}

func (l *EntryLog) Write(data Payload) error {
	l.Size += int64(len(data))

	if false {
		if err := l.sync(); err != nil {
			return err
		}
	}

	return nil
}

func (l *EntryLog) sync() error {
	return nil
}
