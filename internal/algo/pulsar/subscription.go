package main

import (
	"fmt"
	"sync"
)

type SubscriptionType int

const (
	Exclusive SubscriptionType = iota
	Shared
	Failover
)

type Subscription interface {
	Ack(msgID MessageID) error

	Name() string
	Cursor() MessageID
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

func (s *subscription) Ack(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.ackMessages[msgID]; !exists {
		return fmt.Errorf("message %+v not found", msgID)
	}

	delete(s.ackMessages, msgID)
	s.cursor = msgID

	return nil
}
