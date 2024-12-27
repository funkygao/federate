package main

import (
	"sync"
)

type SubscriptionType int

const (
	Exclusive SubscriptionType = iota
	Shared
	Failover
)

type Subscription interface {
	Name() string

	Cursor() MessageID
	MoveCursor(MessageID) error
}

type subscription struct {
	bookKeeper BookKeeper
	topic      *Topic

	name    string
	subType SubscriptionType

	ackMessages map[MessageID]bool
	mu          sync.Mutex

	cursor MessageID
}

func NewSubscription(bk BookKeeper, topic *Topic, name string, subType SubscriptionType) *subscription {
	return &subscription{
		bookKeeper:  bk,
		topic:       topic,
		name:        name,
		subType:     subType,
		ackMessages: make(map[MessageID]bool),
		// cursor 实际上是持久化的，启动时读取
		cursor: MessageID{
			PartitionID: 0,
			LedgerID:    0,
			EntryID:     0,
		},
	}
}

func (s *subscription) Name() string {
	return s.name
}

func (s *subscription) Cursor() MessageID {
	return s.cursor
}

func (s *subscription) MoveCursor(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cursor = msgID
	return nil
}
