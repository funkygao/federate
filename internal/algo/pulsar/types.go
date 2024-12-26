/*
Message Location and Ownership:

[Broker]
 └─ Topic
     └─ Partition
         └─ Ledgers []LedgerID
             └─ [BookKeeper]
                 └─ Ledger (LedgerID)
                     └─ Entry (EntryID)
                          |
                          +--> Stored on [Bookie Nodes]
*/

package main

import (
	"sync/atomic"
	"time"
)

var (
	_ PartitionLB = (*InMemoryBroker)(nil)
	_ BrokerLB    = (*InMemoryBrokerCluster)(nil)
	_ BookieLB    = (*InMemoryBookKeeper)(nil)

	// Ledger IDs are typically managed by BookKeeper, and uniqueness is ensured cluster-wide (often with the help of ZooKeeper).
	_ LedgerIDAllocator = (*InMemoryBookKeeper)(nil)

	_ EntryIDAllocator = (*InMemoryBookie)(nil)
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

func (t *Topic) Subscribe(bk BookKeeper, subscriptionName string, subType SubscriptionType) Subscription {
	sub, exists := t.Subscriptions[subscriptionName]
	if !exists {
		sub = NewInMemorySubscription(bk, t, subscriptionName, subType)
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

type BrokerLB interface {
	selectBroker() Broker
}

type PartitionLB interface {
	selectPartition(msg Message) PartitionID
}

type LedgerIDAllocator interface {
	allocateLedgerID() LedgerID
}

type EntryIDAllocator interface {
	allocateEntryID(LedgerID) EntryID
}

type BookieLB interface {
	selectBookies(ensembleSize int) []Bookie
}
