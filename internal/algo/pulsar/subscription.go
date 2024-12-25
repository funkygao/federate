package main

import (
	"fmt"
	"log"
	"sync"
	"time"
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
	Unsubscribe() error
}

type InMemorySubscription struct {
	bookKeeper  BookKeeper
	topic       *Topic
	name        string
	subType     SubscriptionType
	messages    chan Message
	ackMessages map[MessageID]bool
	cursor      MessageID
	mu          sync.Mutex
}

func NewInMemorySubscription(bk BookKeeper, topic *Topic, name string, subType SubscriptionType) *InMemorySubscription {
	return &InMemorySubscription{
		bookKeeper:  bk,
		topic:       topic,
		name:        name,
		subType:     subType,
		messages:    make(chan Message, 100),
		ackMessages: make(map[MessageID]bool),
	}
}

func (s *InMemorySubscription) Acknowledge(msgID MessageID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.ackMessages[msgID]; !exists {
		return fmt.Errorf("message not found")
	}

	delete(s.ackMessages, msgID)
	s.cursor = msgID

	return nil
}

func (s *InMemorySubscription) Unsubscribe() error {
	// 在实际实现中，这里可能需要清理资源或通知 Broker
	close(s.messages)
	return nil
}

func (s *InMemorySubscription) AddMessage(msg Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	log.Printf("Adding message to subscription: %s", s.name)
	s.ackMessages[msg.ID] = false
	select {
	case s.messages <- msg:
		log.Printf("Message added successfully to subscription: %s", s.name)
	default:
		log.Printf("Warning: Message channel is full for subscription: %s", s.name)
	}
}

func (s *InMemorySubscription) Receive() (Message, error) {
	log.Printf("Attempting to receive message from subscription: %s", s.name)
	select {
	case msg := <-s.messages:
		log.Printf("Message received from subscription: %s", s.name)
		return msg, nil
	case <-time.After(5 * time.Second):
		return Message{}, fmt.Errorf("timeout")
	}
}
