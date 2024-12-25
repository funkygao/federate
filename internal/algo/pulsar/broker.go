package main

import (
	"sync"
)

type Broker interface {
	Publish(topic string, msg Message) error
	Subscribe(topic, subscriptionName string, subType SubscriptionType) (Subscription, error)
}

type PulsarBroker struct {
	bookKeeper BookKeeper
	topics     map[string]*TopicManager
	mu         sync.Mutex
}

func NewPulsarBroker(bk BookKeeper) *PulsarBroker {
	return &PulsarBroker{
		bookKeeper: bk,
		topics:     make(map[string]*TopicManager),
	}
}

func (b *PulsarBroker) Publish(topic string, msg Message) error {
	b.mu.Lock()
	tm, ok := b.topics[topic]
	if !ok {
		tm = NewTopicManager(b.bookKeeper)
		b.topics[topic] = tm
	}
	b.mu.Unlock()
	return tm.Publish(msg)
}

func (b *PulsarBroker) Subscribe(topic, subscriptionName string, subType SubscriptionType) (Subscription, error) {
	b.mu.Lock()
	tm, ok := b.topics[topic]
	if !ok {
		tm = NewTopicManager(b.bookKeeper)
		b.topics[topic] = tm
	}
	b.mu.Unlock()
	return tm.Subscribe(subscriptionName, subType)
}
