/*
Message Location and Ownership:

[Broker]
 └─ Topic
     └─ Partition
         └─ TimeSegment (TimeSegmentID)
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

type PartitionID int
type TimeSegmentID int64

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
	PartitionID   PartitionID
	TimeSegmentID TimeSegmentID
	LedgerID      LedgerID
	EntryID       EntryID
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

func (m *Message) ReadyTime() time.Time {
	return time.Now().Add(m.Delay)
}

type Topic struct {
	Name          string
	Partitions    map[PartitionID]*Partition
	Subscriptions map[string]Subscription
}

func (t *Topic) Subscribe(BookKeeper bk, subscriptionName string, subType SubscriptionType) Subscription {
	sub, exists := t.Subscriptions[subscriptionName]
	if !exists {
		sub = NewInMemorySubscription(bk, t, subscriptionName, subType)
		topic.Subscriptions[subscriptionName] = sub
	}
	return sub
}

func (t *Topic) GetParttion(partitionID PartitionID) *Partition {
	partition, exists := t.Partitions[partitionID]
	if !exists {
		partition = &Partition{
			ID:           partitionID,
			TimeSegments: make(map[TimeSegmentID]*TimeSegment),
		}
		t.Partitions[partitionID] = partition
	}
	return partition
}

// Partition 表示 Topic 的一个分区，比 Kafka 增加更细粒度的 TimeSegment
type Partition struct {
	ID           PartitionID
	TimeSegments map[TimeSegmentID]*TimeSegment
}

// TimeSegment 表示一个时间段内的数据，一个 TimeSegment 必须存放在一个 Bookie
type TimeSegment struct {
	ID      TimeSegmentID
	Ledgers []LedgerID
}
