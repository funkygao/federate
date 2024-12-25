package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Responsible for message routing, subscription management, and interacting with the storage layer (BookKeeper) to persist messages.
type Broker interface {
	CreateTopic(name string) (*Topic, error)
	GetTopic(name string) (*Topic, error)

	CreateProducer(topic string) (Producer, error)
	Subscribe(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)

	Publish(topicName string, msg Message) error
}

type InMemoryBroker struct {
	// Responsible for distributing ledgers and entries across available bookies for load balancing and fault tolerance
	bookKeeper BookKeeper

	topics map[string]*Topic
	mu     sync.RWMutex

	delayQueue *DelayQueue
}

func NewInMemoryBroker(bk BookKeeper) *InMemoryBroker {
	broker := &InMemoryBroker{
		bookKeeper: bk,
		topics:     make(map[string]*Topic),
		delayQueue: NewDelayQueue(),
	}
	go broker.processDelayedMessages()
	return broker
}

func (b *InMemoryBroker) CreateTopic(name string) (*Topic, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.topics[name]; exists {
		return nil, fmt.Errorf("topic already exists")
	}

	topic := &Topic{
		Name:          name,
		Partitions:    make(map[PartitionID]*Partition),
		Subscriptions: make(map[string]*InMemorySubscription),
	}
	b.topics[name] = topic
	return topic, nil
}

func (b *InMemoryBroker) GetTopic(name string) (*Topic, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	topic, exists := b.topics[name]
	if !exists {
		return nil, fmt.Errorf("topic not found")
	}
	return topic, nil
}

func (b *InMemoryBroker) CreateProducer(topic string) (Producer, error) {
	t, err := b.GetTopic(topic)
	if err != nil {
		return nil, err
	}

	return NewInMemoryProducer(b, t), nil
}

func (b *InMemoryBroker) Subscribe(topicName, subscriptionName string, subType SubscriptionType) (Consumer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.Printf("Attempting to subscribe to topic: %s with subscription: %s", topicName, subscriptionName)

	topic, exists := b.topics[topicName]
	if !exists {
		return nil, fmt.Errorf("topic not found")
	}

	sub, exists := topic.Subscriptions[subscriptionName]
	if !exists {
		log.Printf("Creating new subscription: %s for topic: %s", subscriptionName, topicName)
		sub = NewInMemorySubscription(b.bookKeeper, topic, subscriptionName, subType)
		topic.Subscriptions[subscriptionName] = sub
	}

	log.Printf("Subscription created/found: %s for topic: %s", subscriptionName, topicName)
	return NewInMemoryConsumer(sub), nil
}

func (b *InMemoryBroker) processDelayedMessages() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			now := time.Now()
			for {
				msg, ok := b.delayQueue.Poll()
				if !ok {
					break
				}
				if msg.Timestamp.After(now) {
					// 消息还没有准备好，放回队列
					b.delayQueue.Add(msg, msg.Timestamp)
					break
				}
				// 消息已经准备好，发布到相应的主题
				err := b.Publish(msg.Topic, msg)
				if err != nil {
					log.Printf("Error publishing delayed message: %v", err)
				}
			}
		}
	}
}

func (b *InMemoryBroker) Publish(topicName string, msg Message) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.Printf("Attempting to publish message to topic: %s", topicName)

	topic, exists := b.topics[topicName]
	if !exists {
		return fmt.Errorf("topic not found")
	}

	// 简化：使用单一分区
	partitionID := PartitionID(0)
	partition, exists := topic.Partitions[partitionID]
	if !exists {
		partition = &Partition{
			ID:           partitionID,
			TimeSegments: make(map[TimeSegmentID]*TimeSegment),
		}
		topic.Partitions[partitionID] = partition
	}

	// Get or create TimeSegment
	timeSegment, err := b.getOrCreateTimeSegment(partition)
	if err != nil {
		return err
	}

	// Get or create Ledger
	ledger, err := b.getOrCreateLedger(timeSegment)
	if err != nil {
		return err
	}

	// 添加消息到 Ledger
	entryID, err := ledger.AddEntry(msg.Content)
	if err != nil {
		return err
	}

	msg.Topic = topicName
	msg.ID = MessageID{
		PartitionID:   partitionID,
		TimeSegmentID: timeSegment.ID,
		LedgerID:      ledger.GetID(),
		EntryID:       entryID,
	}

	// 处理延迟消息
	if msg.Delay > 0 {
		b.delayQueue.Add(msg, time.Now().Add(msg.Delay))
	} else {
		b.deliverMessage(topicName, msg)
	}

	log.Printf("Message published successfully to topic: %s", topicName)
	log.Printf("Delivering message to %d subscriptions", len(topic.Subscriptions))

	for subName, sub := range topic.Subscriptions {
		log.Printf("Delivering message to subscription: %s", subName)
		sub.AddMessage(msg)
	}

	return nil
}

func (b *InMemoryBroker) deliverMessage(topicName string, msg Message) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	topic, exists := b.topics[topicName]
	if !exists {
		log.Printf("Warning: Topic %s not found for message delivery", topicName)
		return
	}

	for _, sub := range topic.Subscriptions {
		sub.AddMessage(msg)
	}
}

func (b *InMemoryBroker) getOrCreateLedger(timeSegment *TimeSegment) (Ledger, error) {
	var ledger Ledger
	var err error

	if len(timeSegment.Ledgers) == 0 {
		ledger, err = b.bookKeeper.CreateLedger()
		if err != nil {
			return nil, err
		}
		timeSegment.Ledgers = append(timeSegment.Ledgers, ledger.GetID())
	} else {
		lastLedgerID := timeSegment.Ledgers[len(timeSegment.Ledgers)-1]
		ledger, err = b.bookKeeper.OpenLedger(lastLedgerID)
		if err != nil {
			return nil, err
		}
	}

	return ledger, nil
}

func (b *InMemoryBroker) getOrCreateTimeSegment(partition *Partition) (*TimeSegment, error) {
	currentTimeSegmentID := b.allocateTimeSegmentID()

	b.mu.Lock()
	defer b.mu.Unlock()

	timeSegment, exists := partition.TimeSegments[currentTimeSegmentID]
	if !exists {
		timeSegment = &TimeSegment{
			ID:      currentTimeSegmentID,
			Ledgers: []LedgerID{},
		}
		partition.TimeSegments[currentTimeSegmentID] = timeSegment
	}

	return timeSegment, nil
}

func (b *InMemoryBroker) allocateTimeSegmentID() TimeSegmentID {
	return TimeSegmentID(time.Now().Unix() / (60 * 60)) // Segments per hour
}
