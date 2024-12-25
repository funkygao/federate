package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

type InMemoryBroker struct {
	bookKeeper   BookKeeper
	topicManager *TopicManager // TODO not used

	topics map[string]*Topic
	mu     sync.RWMutex

	delayQueue *DelayQueue
}

func NewInMemoryBroker(bk BookKeeper) *InMemoryBroker {
	broker := &InMemoryBroker{
		bookKeeper:   bk,
		topicManager: NewTopicManager(),
		topics:       make(map[string]*Topic),
		delayQueue:   NewDelayQueue(),
	}
	go broker.processDelayedMessages()
	return broker
}

func (b *InMemoryBroker) CreateTopic(name string) (*Topic, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.topics[name]; exists {
		return nil, &BrokerError{Op: "CreateTopic", Err: fmt.Errorf("topic already exists")}
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
		return nil, &BrokerError{Op: "GetTopic", Err: fmt.Errorf("topic not found")}
	}
	return topic, nil
}

func (b *InMemoryBroker) DeleteTopic(name string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.topics[name]; !exists {
		return &BrokerError{Op: "DeleteTopic", Err: fmt.Errorf("topic not found")}
	}

	delete(b.topics, name)
	return nil
}

func (b *InMemoryBroker) CreateProducer(topic string) (Producer, error) {
	t, err := b.GetTopic(topic)
	if err != nil {
		return nil, &BrokerError{Op: "CreateProducer", Err: err}
	}

	return NewInMemoryProducer(b, t), nil
}

func (b *InMemoryBroker) Subscribe(topicName, subscriptionName string, subType SubscriptionType) (Consumer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.Printf("Attempting to subscribe to topic: %s with subscription: %s", topicName, subscriptionName)

	topic, exists := b.topics[topicName]
	if !exists {
		return nil, &BrokerError{Op: "Subscribe", Err: fmt.Errorf("topic not found")}
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
					// 队列为空或没有准备好的消息
					break
				}
				if msg.Timestamp.After(now) {
					// 消息还没有准备好，放回队列
					b.delayQueue.Add(msg, msg.Timestamp)
					break
				}
				// 消息已经准备好，发布到相应的主题
				err := b.publishMessage(msg.Topic, msg)
				if err != nil {
					log.Printf("Error publishing delayed message: %v", err)
				}
			}
		}
	}
}

func (b *InMemoryBroker) publishMessage(topicName string, msg Message) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	log.Printf("Attempting to publish message to topic: %s", topicName)

	topic, exists := b.topics[topicName]
	if !exists {
		return &BrokerError{Op: "publishMessage", Err: fmt.Errorf("topic not found")}
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

	// 获取或创建当前的 TimeSegment
	currentTime := TimeSegmentID(time.Now().Unix() / (60 * 60)) // 每小时一个 TimeSegment
	timeSegment, exists := partition.TimeSegments[currentTime]
	if !exists {
		timeSegment = &TimeSegment{
			ID:      currentTime,
			Ledgers: []LedgerID{},
		}
		partition.TimeSegments[currentTime] = timeSegment
	}

	// 创建新的 Ledger（如果需要）
	var ledgerID LedgerID
	if len(timeSegment.Ledgers) == 0 {
		var err error
		ledgerID, err = b.bookKeeper.CreateLedger()
		if err != nil {
			return &BrokerError{Op: "publishMessage", Err: err}
		}
		timeSegment.Ledgers = append(timeSegment.Ledgers, ledgerID)
	} else {
		ledgerID = timeSegment.Ledgers[len(timeSegment.Ledgers)-1]
	}

	// 添加消息到 BookKeeper
	ledger, err := b.bookKeeper.OpenLedger(ledgerID)
	if err != nil {
		return &BrokerError{Op: "publishMessage", Err: err}
	}

	entryID, err := ledger.AddEntry(msg.Content)
	if err != nil {
		return &BrokerError{Op: "publishMessage", Err: err}
	}

	msg.Topic = topicName
	msg.ID = MessageID{
		PartitionID:   partitionID,
		TimeSegmentID: currentTime,
		LedgerID:      ledgerID,
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
