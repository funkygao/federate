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
	CreateConsumer(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)

	Publish(msg Message) error
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

func (b *InMemoryBroker) CreateProducer(topicName string) (Producer, error) {
	topic, err := b.GetTopic(topicName)
	if err != nil {
		return nil, err
	}

	return NewInMemoryProducer(b, topic), nil
}

func (b *InMemoryBroker) CreateConsumer(topicName, subscriptionName string, subType SubscriptionType) (Consumer, error) {
	topic, err := b.GetTopic(topicName)
	if err != nil {
		return nil, err
	}

	sub := topic.Subscribe(b.bookKeeper, subscriptionName, subType)
	return NewInMemoryConsumer(sub), nil
}

// TODO Broker hould have Publish?
func (b *InMemoryBroker) Publish(msg Message) error {
	topic, err := b.GetTopic(msg.Topic)
	if err != nil {
		return nil, err
	}

	partition, timeSegment, ledger, err := b.routeMessage(msg)
	if err != nil {
		return err
	}

	// TODO replica write
	entryID, err := ledger.AddEntry(msg.Content)
	if err != nil {
		return err
	}

	msg.ID = MessageID{
		PartitionID:   partition.ID,
		TimeSegmentID: timeSegment.ID,
		LedgerID:      ledger.GetID(),
		EntryID:       entryID,
	}

	if msg.IsDelay() {
		b.delayQueue.Add(msg)
	} else {
		// TODO kill deliverMessage, should poll from BookKeeper
		b.deliverMessage(msg)
	}

	// TODO kill
	for subName, sub := range topic.Subscriptions {
		log.Printf("Delivering message to subscription: %s", subName)
		sub.AddMessage(msg)
	}

	return nil
}

func (b *InMemoryBroker) routeMessage(msg Message) (partition Partition, timeSegment TimeSegment, ledger Ledger, err error) {
	partition = topic.GetParttion(b.selectPartition(msg))

	timeSegment, err = b.getOrCreateTimeSegment(partition)
	if err != nil {
		return
	}

	ledger, err = b.getOrCreateLedger(timeSegment)
	if err != nil {
		return
	}
	return
}

func (b *InMemoryBroker) deliverMessage(msg Message) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	topic, exists := b.topics[msg.Topic]
	if !exists {
		log.Printf("Warning: Topic %s not found for message delivery", msg.Topic)
		return
	}

	for _, sub := range topic.Subscriptions {
		sub.AddMessage(msg)
	}
}

func (b *InMemoryBroker) processDelayedMessages() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case now := <-ticker.C:
			for {
				// TODO Peek，就不用放回队列了
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

func (b *InMemoryBroker) selectPartition(msg Message) PartitionID {
	return PartitionID(0)
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
