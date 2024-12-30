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

	// Broker is owner of topics
	CreateTopic(name string, ledgerOption LedgerOption) (*Topic, error)
	GetTopic(name string) (*Topic, error)

	CreateProducer(topic string) (Producer, error)
	CreateConsumer(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)

	Publish(Message) error
	Receive(topic string, subscriptionName string) (Message, error)
	Ack(topic string, subscriptionName string, msgID MessageID) error
}

type broker struct {
	info BrokerInfo

	// BookKeeper 暴露给 Broker 的RPC服务
	bkClient BookKeeper

	zk ZooKeeper

	topics map[string]*Topic
	mu     sync.RWMutex

	delayQueue *DelayQueue

	// 消费尾部消息时，不必访问 BK
	tailCache TailCache

	cursorStore CursorStore
}

func NewBroker(bk BookKeeper) *broker {
	broker := &broker{
		bkClient:    bk,
		topics:      make(map[string]*Topic),
		delayQueue:  NewDelayQueue(),
		zk:          getZooKeeper(),
		cursorStore: NewCursorStore(bk),
	}
	go broker.processDelayedMessages()
	go broker.manageLedgerRetention()
	go broker.rebalance()
	return broker
}

func (b *broker) Start() error {
	b.info = BrokerInfo{
		ID:   "1",
		Host: "localhost",
		Port: 9988,
	}

	b.zk.RegisterBroker(b.info)

	log.Printf("Broker%+v started", b.info)
	return nil
}

func (b *broker) CreateTopic(name string, ledgerOption LedgerOption) (*Topic, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.topics[name]; exists {
		return nil, fmt.Errorf("topic already exists")
	}

	topic := &Topic{
		Name:          name,
		LedgerOption:  ledgerOption,
		Partitions:    make(map[PartitionID]*Partition),
		Subscriptions: make(map[string]Subscription),
	}
	b.topics[name] = topic

	b.zk.RegisterTopic(*topic)

	log.Printf("%s CreateTopic(%s)", b.logIdent(), name)
	return topic, nil
}

func (b *broker) GetTopic(name string) (*Topic, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	topic, exists := b.topics[name]
	if !exists {
		return nil, fmt.Errorf("topic not found")
	}

	return topic, nil
}

func (b *broker) CreateProducer(topicName string) (Producer, error) {
	topic, err := b.GetTopic(topicName)
	if err != nil {
		return nil, err
	}

	log.Printf("%s CreateProducer for topic: %s", b.logIdent(), topicName)
	return NewProducer(b, topic), nil
}

func (b *broker) CreateConsumer(topicName, subscriptionName string, subType SubscriptionType) (Consumer, error) {
	topic, err := b.GetTopic(topicName)
	if err != nil {
		return nil, err
	}

	sub := topic.Subscribe(b.bkClient, subscriptionName, subType)

	log.Printf("%s CreateConsumerfor topic: %s, subscription: %+v", b.logIdent(), topicName, sub)
	return NewConsumer(b, topic, sub), nil
}

func (b *broker) Publish(msg Message) error {
	topic, err := b.GetTopic(msg.Topic)
	if err != nil {
		return err
	}

	partition, ledger, err := b.routeMessage(topic, msg)
	if err != nil {
		return err
	}

	log.Printf("%s Publish(%s), routing info: {PartitionID:%d, LedgerID:%d}", b.logIdent(), msg.Content, partition.ID, ledger.GetLedgerID())

	entryID, err := ledger.AddEntry(msg.Content, b.bkClient.LedgerOption(ledger.GetLedgerID()))
	if err != nil {
		return err
	}

	msg.ID = MessageID{
		PartitionID: partition.ID,
		LedgerID:    ledger.GetLedgerID(),
		EntryID:     entryID,
	}

	// Append Tail Cache
	if b.tailCache != nil {
		b.tailCache.Put(*topic, msg)
	}

	if msg.IsDelay() {
		log.Printf("%s Publish got delay message, ready at: %v", b.logIdent(), msg.ReadyTime())
		b.delayQueue.Add(msg)
	}

	return nil
}

func (b *broker) Ack(topicName string, subscriptionName string, msgID MessageID) error {
	_, sub, err := b.getSubscription(topicName, subscriptionName)
	if err != nil {
		return err
	}

	// 更新游标
	err = sub.MoveCursor(msgID)
	if err != nil {
		return err
	}

	// 异步持久化游标
	go func() {
		if err := b.cursorStore.SaveCursor(topicName, subscriptionName, msgID); err != nil {
			log.Printf("Failed to persist cursor: %v", err)
		}
	}()

	return nil
}

func (b *broker) Receive(topicName string, subscriptionName string) (Message, error) {
	topic, sub, err := b.getSubscription(topicName, subscriptionName)
	if err != nil {
		return Message{}, err
	}

	partition := topic.GetPartition(sub.Cursor().PartitionID)

	for {
		cursor := sub.Cursor()
		if cursor.LedgerID == 0 {
			// 初始化游标
			newCursor, err := b.cursorStore.LoadCursor(topicName, subscriptionName, partition)
			if err != nil {
				return Message{}, err
			}
			sub.MoveCursor(newCursor)
			cursor = newCursor
		}

		ledger, err := b.bkClient.OpenLedger(cursor.LedgerID)
		if err != nil {
			return Message{}, fmt.Errorf("failed to open ledger: %v", err)
		}

		msg, found, err := b.tryReadNextEntry(ledger, sub)
		if err != nil {
			return Message{}, err
		}
		if found {
			sub.MoveCursor(msg.ID)
			return msg, nil
		}

		// 如果当前 ledger 读完，尝试移动到下一个 ledger
		nextLedgerIndex := partition.IndexOfLedger(cursor.LedgerID) + 1
		if nextLedgerIndex < len(partition.Ledgers) {
			newCursor := MessageID{
				PartitionID: cursor.PartitionID,
				LedgerID:    partition.Ledgers[nextLedgerIndex],
				EntryID:     0,
			}
			sub.MoveCursor(newCursor)
			continue
		}

		// 如果没有更多的 ledger，等待新消息
		time.Sleep(100 * time.Millisecond)
	}
}

func (b *broker) getSubscription(topicName, subscriptionName string) (*Topic, Subscription, error) {
	topic, err := b.GetTopic(topicName)
	if err != nil {
		return nil, nil, err
	}

	sub := topic.GetSubscription(subscriptionName)
	if sub == nil {
		return nil, nil, fmt.Errorf("subscription not found")
	}

	return topic, sub, nil
}

func (b *broker) tryReadNextEntry(ledger Ledger, sub Subscription) (Message, bool, error) {
	lastConfirmed := ledger.GetLastAddConfirmed()
	nextEntryID := sub.Cursor().EntryID + 1

	if nextEntryID <= lastConfirmed {
		payload, err := ledger.ReadEntry(nextEntryID)
		if err != nil {
			return Message{}, false, fmt.Errorf("failed to read entry: %v", err)
		}

		msg := Message{
			ID: MessageID{
				PartitionID: sub.Cursor().PartitionID,
				LedgerID:    sub.Cursor().LedgerID,
				EntryID:     nextEntryID,
			},
			Content: payload,
		}

		return msg, true, nil
	}

	return Message{}, false, nil
}

func (b *broker) routeMessage(topic *Topic, msg Message) (partition *Partition, ledger Ledger, err error) {
	// 先分区
	partition = topic.GetPartition(b.selectPartition(topic, msg))

	// 再分ledger，ledger 是 pulsar 比 Kafka 更灵活的最根本设计
	ledger, err = b.getOrCreateLedger(topic, partition)
	return
}

func (b *broker) selectPartition(topic *Topic, msg Message) PartitionID {
	// 实际上类似 Kafka Partitioner，可能根据消息key分区
	return PartitionID(0)
}

func (b *broker) getOrCreateLedger(topic *Topic, partition *Partition) (Ledger, error) {
	var ledger Ledger
	var err error

	if len(partition.Ledgers) == 0 {
		// Initialize
		ledger, err = b.createNewLedger(topic, partition)
		if err != nil {
			return nil, err
		}
	} else {
		// Get the last ledger
		lastLedgerID := partition.Ledgers[len(partition.Ledgers)-1]
		ledger, err = b.bkClient.OpenLedger(lastLedgerID)
		if err != nil {
			return nil, err
		}

		// Check if we need to roll over to a new ledger
		if b.shouldRolloverLedger(ledger) {
			// Close the current ledger if necessary
			ledger, err = b.createNewLedger(topic, partition)
			if err != nil {
				return nil, err
			}
		}
	}

	return ledger, nil
}

func (b *broker) createNewLedger(topic *Topic, partition *Partition) (Ledger, error) {
	ledger, err := b.bkClient.CreateLedger(topic.LedgerOption)
	if err != nil {
		return nil, err
	}

	partition.Ledgers = append(partition.Ledgers, ledger.GetLedgerID())
	return ledger, nil
}

func (b *broker) shouldRolloverLedger(ledger Ledger) bool {
	// e,g. based on the number of entries or age of the ledger
	return false
}

func (b *broker) logIdent() string {
	return "Broker[" + b.info.ID + "]"
}
