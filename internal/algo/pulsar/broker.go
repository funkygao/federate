package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// Broker: Manages topics, producers, consumers, message routing, and interacts with the storage layer.
type Broker interface {
	Start() error

	CreateTopic(name string) (*Topic, error)
	GetTopic(name string) (*Topic, error)

	CreateProducer(topic string) (Producer, error)
	CreateConsumer(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)

	Publish(msg Message) error
}

type InMemoryBroker struct {
	// BookKeeper 暴露给 Broker 的RPC服务
	bookKeeper BookKeeper

	zk ZooKeeper

	topics map[string]*Topic
	mu     sync.RWMutex

	delayQueue *DelayQueue
}

func NewInMemoryBroker(bk BookKeeper) *InMemoryBroker {
	broker := &InMemoryBroker{
		bookKeeper: bk,
		topics:     make(map[string]*Topic),
		delayQueue: NewDelayQueue(),
		zk:         getZooKeeper(),
	}
	go broker.processDelayedMessages()
	return broker
}

func (b *InMemoryBroker) Start() error {
	info := BrokerInfo{
		ID:   "1",
		Host: "localhost",
		Port: 9988,
	}
	b.zk.RegisterBroker(info)
	return nil
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
		Subscriptions: make(map[string]Subscription),
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

func (b *InMemoryBroker) Publish(msg Message) error {
	topic, err := b.GetTopic(msg.Topic)
	if err != nil {
		return err
	}

	partition, ledger, err := b.routeMessage(topic, msg)
	if err != nil {
		return err
	}

	entryID, err := ledger.AddEntry(msg.Content, b.bookKeeper.LedgerOption(ledger.GetLedgerID()))
	if err != nil {
		return err
	}

	msg.ID = MessageID{
		PartitionID: partition.ID,
		LedgerID:    ledger.GetLedgerID(),
		EntryID:     entryID,
	}

	if msg.IsDelay() {
		b.delayQueue.Add(msg)
	}

	return nil
}

func (b *InMemoryBroker) routeMessage(topic *Topic, msg Message) (partition *Partition, ledger Ledger, err error) {
	partition = topic.GetPartition(b.selectPartition(msg))

	ledger, err = b.getOrCreateLedger(partition)
	if err != nil {
		return
	}
	return
}

func (b *InMemoryBroker) processDelayedMessages() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// 一个时间周期内处理尽可能多的延迟消息
			for {
				msg, hasDueMsg := b.delayQueue.Poll()
				if !hasDueMsg {
					break
				}

				// 消息已经准备好，发布到相应的主题
				if err := b.Publish(msg); err != nil {
					log.Printf("Error publishing delayed message: %v", err)
				}
			}
		}
	}
}

func (b *InMemoryBroker) selectPartition(msg Message) PartitionID {
	return PartitionID(0)
}

func (b *InMemoryBroker) getOrCreateLedger(partition *Partition) (Ledger, error) {
	var ledger Ledger
	var err error

	if len(partition.Ledgers) == 0 {
		// getOrCreateLedger
		ledger, err = b.createNewLedger(partition)
		if err != nil {
			return nil, err
		}
	} else {
		// Get the last ledger
		lastLedgerID := partition.Ledgers[len(partition.Ledgers)-1]
		ledger, err = b.bookKeeper.OpenLedger(lastLedgerID)
		if err != nil {
			return nil, err
		}

		// Check if we need to roll over to a new ledger
		if b.shouldRolloverLedger(ledger) {
			// Close the current ledger if necessary
			ledger, err = b.createNewLedger(partition)
			if err != nil {
				return nil, err
			}
		}
	}

	return ledger, nil
}

func (b *InMemoryBroker) createNewLedger(partition *Partition) (Ledger, error) {
	ledger, err := b.bookKeeper.CreateLedger(LedgerOption{
		EnsembleSize: 3,
		WriteQuorum:  2,
		AckQuorum:    2,
	})
	if err != nil {
		return nil, err
	}

	partition.Ledgers = append(partition.Ledgers, ledger.GetLedgerID())
	return ledger, nil
}

func (b *InMemoryBroker) shouldRolloverLedger(ledger Ledger) bool {
	// For example, based on the number of entries or age of the ledger
	return false
}
