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

type LedgerID int64
type EntryID int64
type PartitionID int

// TimeSegmentID 唯一标识一个时间段
type TimeSegmentID int64

type Payload []byte

// MessageID 唯一标识一条消息在 Pulsar 系统中的位置
//
//	结构:
//	    +--------------+--------------+----------------+-----------+
//	    | PartitionID  | TimeSegmentID|    LedgerID    |  EntryID  |
//	    +--------------+--------------+----------------+-----------+
type MessageID struct {
	PartitionID   PartitionID
	TimeSegmentID TimeSegmentID
	LedgerID      LedgerID
	EntryID       EntryID
}

type Message struct {
	ID        MessageID
	Content   Payload
	Timestamp time.Time
	Delay     time.Duration
}

// Partition 表示 Topic 的一个分区，包含多个 TimeSegment
//
//	+-------------+
//	| Partition   |
//	+-------------+
//	| ID          |
//	| TimeSegments|
//	+-------------+
type Partition struct {
	ID           PartitionID
	TimeSegments []TimeSegment
}

// TimeSegment 表示一个时间段内的数据，包含多个 Ledger
//
//	+-------------+
//	| TimeSegment |
//	+-------------+
//	| ID          |
//	| Ledgers     |
//	+-------------+
type TimeSegment struct {
	ID      TimeSegmentID
	Ledgers []Ledger
}

// Ledger 表示 BookKeeper 中的一个账本，包含一系列有序的 Entry
//
//	+----------+
//	| Ledger   |
//	+----------+
//	| ID       |
//	| Entries  |
//	+----------+
type Ledger struct {
	ID      LedgerID
	Entries map[EntryID]Payload
}
