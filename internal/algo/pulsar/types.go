/*
Message Location and Ownership:

[Broker]
 └─ Topic
     └─ Partition
         └─ [BookKeeper]
             └─ Ledger
                 └─ [Bookie]
                      └─ Entry
*/

package main

import (
	"sync/atomic"
	"time"
)

var (
	prodEnv = false

	_ PartitionLB       = (*broker)(nil)
	_ BookieLB          = (*bookKeeper)(nil)
	_ LedgerIDAllocator = (*bookKeeper)(nil)
	_ EntryIDAllocator  = (*ledger)(nil)
)

type PartitionID int

type LedgerID int64

func (id *LedgerID) Next() LedgerID {
	return LedgerID(atomic.AddInt64((*int64)(id), 1))
}

type EntryID int64

func (id *EntryID) Next() EntryID {
	return EntryID(atomic.AddInt64((*int64)(id), 1))
}

// MessageID 唯一标识一条消息在 Pulsar 系统中的位置
type MessageID struct {
	PartitionID PartitionID
	LedgerID    LedgerID
	EntryID     EntryID
}

// Payload 表示消息的实际内容
type Payload []byte

func (p Payload) Size() int64 {
	return int64(len(p))
}

// Message 表示一条消息
type Message struct {
	ID        MessageID
	Topic     string
	Content   Payload
	Timestamp time.Time
	Delay     time.Duration
}

func (m *Message) IsDelay() bool {
	return m.Delay > 0
}

func (m *Message) IsDue() bool {
	return !time.Now().Before(m.ReadyTime())
}

func (m *Message) ReadyTime() time.Time {
	return time.Now().Add(m.Delay)
}

func (m Message) String() string {
	return string(m.Content)
}

type Topic struct {
	Name          string
	Partitions    map[PartitionID]*Partition
	Subscriptions map[string]Subscription
}

func (t *Topic) GetSubscription(subscriptionName string) Subscription {
	return t.Subscriptions[subscriptionName]
}

func (t *Topic) Subscribe(bk BookKeeper, subscriptionName string, subType SubscriptionType) Subscription {
	sub, exists := t.Subscriptions[subscriptionName]
	if !exists {
		sub = NewSubscription(bk, t, subscriptionName, subType)
		t.Subscriptions[subscriptionName] = sub
	}
	return sub
}

func (t *Topic) GetPartition(partitionID PartitionID) *Partition {
	partition, exists := t.Partitions[partitionID]
	if !exists {
		partition = &Partition{
			ID:      partitionID,
			Ledgers: []LedgerID{},
		}
		t.Partitions[partitionID] = partition
	}
	return partition
}

// Partition 表示 Topic 的一个分区，比 Kafka 增加更细粒度的 Ledger
type Partition struct {
	ID      PartitionID
	Ledgers []LedgerID
}

func (p *Partition) IndexOfLedger(ledgerID LedgerID) int {
	for i, id := range p.Ledgers {
		if id == ledgerID {
			return i
		}
	}
	return -1
}

type PartitionLB interface {
	selectPartition(*Topic, Message) PartitionID
}

type LedgerIDAllocator interface {
	allocateLedgerID() LedgerID
}

type EntryIDAllocator interface {
	allocateEntryID() EntryID
}

type BookieLB interface {
	selectBookies(LedgerID, LedgerOption) []Bookie
}
