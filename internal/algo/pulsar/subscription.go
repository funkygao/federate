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
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
	AddConsumer(consumer Consumer) error
	RemoveConsumer(consumer Consumer) error
}

type PulsarSubscription struct {
	bookKeeper  BookKeeper
	subType     SubscriptionType
	cursor      MessageID
	newMessages chan Message // 改为 Message 类型
	consumers   []Consumer
	mu          sync.Mutex
}

func NewPulsarSubscription(bk BookKeeper, subType SubscriptionType) *PulsarSubscription {
	return &PulsarSubscription{
		bookKeeper:  bk,
		subType:     subType,
		cursor:      MessageID{},
		newMessages: make(chan Message, 100),
		consumers:   make([]Consumer, 0),
	}
}

func (s *PulsarSubscription) Receive() (Message, error) {
	msg := <-s.newMessages
	return msg, nil
}

func (s *PulsarSubscription) Acknowledge(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cursor = msgID
	return nil
}

func (s *PulsarSubscription) AddConsumer(consumer Consumer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.consumers = append(s.consumers, consumer)
	return nil
}

func (s *PulsarSubscription) RemoveConsumer(consumer Consumer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, c := range s.consumers {
		if c == consumer {
			s.consumers = append(s.consumers[:i], s.consumers[i+1:]...)
			break
		}
	}
	return nil
}

func (s *PulsarSubscription) NotifyNewMessage(msg Message) {
	s.newMessages <- msg
}
