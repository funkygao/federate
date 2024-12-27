package main

import "fmt"

// 保存在 BK 一个特殊 ledger里，就像 Kafka 保存在一个特殊 topic
type CursorStore interface {
	SaveCursor(topic string, subscription string, cursor MessageID) error
	LoadCursor(topic string, subscription string, partition *Partition) (MessageID, error)
}

type cursorStore struct {
	bkClient BookKeeper
}

func NewCursorStore(bk BookKeeper) CursorStore {
	return &cursorStore{bk}
}

func (cs *cursorStore) SaveCursor(topic string, subscription string, cursor MessageID) error {
	return nil
}

func (cs *cursorStore) LoadCursor(topic string, subscription string, partition *Partition) (MessageID, error) {
	// 如果没有现有的游标，初始化为第一个可用的 ledger 的开始
	if len(partition.Ledgers) == 0 {
		return MessageID{}, fmt.Errorf("no ledgers available")
	}

	return MessageID{
		PartitionID: partition.ID,
		LedgerID:    partition.Ledgers[0],
		EntryID:     0,
	}, nil
}
