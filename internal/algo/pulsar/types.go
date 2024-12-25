/*
Pulsar 核心概念关系图:

Topic
  |
  +-- Partition 1
  |     |
  |     +-- TimeSegment 1
  |     |     |
  |     |     +-- Ledger 1
  |     |     |     |
  |     |     |     +-- Entry 1 (Message)
  |     |     |     +-- Entry 2 (Message)
  |     |     |     +-- ...
  |     |     |
  |     |     +-- Ledger 2
  |     |     +-- ...
  |     |
  |     +-- TimeSegment 2
  |     +-- ...
  |
  +-- Partition 2
  +-- ...

                存储在
Ledger 1 ------------------------+
Ledger 2 -----------------+      |
Ledger 3 ----------+      |      |
                   |      |      |
                   v      v      v
              +----------+  +----------+  +----------+
              | Bookie 1 |  | Bookie 2 |  | Bookie 3 |
              +----------+  +----------+  +----------+

消息定位: Topic -> Partition -> TimeSegment -> Ledger -> Entry
消息 ID:  PartitionID.TimeSegmentID.LedgerID.EntryID

Broker: 管理 Topic 和 Partition，处理消息的发布和订阅
Bookie: BookKeeper 的存储节点，负责存储 Ledger 数据
*/

package main

import "time"

type PartitionID int
type TimeSegmentID int64
type LedgerID int64
type EntryID int64

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

type Topic struct {
	Name          string
	Partitions    map[PartitionID]*Partition
	Subscriptions map[string]*InMemorySubscription
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
