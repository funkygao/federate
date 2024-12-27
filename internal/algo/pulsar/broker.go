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
	CreateTopic(name string) (*Topic, error)
	GetTopic(name string) (*Topic, error)

	CreateProducer(topic string) (Producer, error)
	CreateConsumer(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)

	Publish(msg Message) error
	Receive(topic string, subscriptionName string) (Message, error)
}

type TailCache interface {
	Put(Topic, Message)
	Get(Topic) Message
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
}

func NewBroker(bk BookKeeper) *broker {
	broker := &broker{
		bkClient:   bk,
		topics:     make(map[string]*Topic),
		delayQueue: NewDelayQueue(),
		zk:         getZooKeeper(),
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

func (b *broker) CreateTopic(name string) (*Topic, error) {
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

	log.Printf("%s Publish, routing info: {partition: %+v, ledger %+v}", b.logIdent(), partition, ledger)

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

func (b *broker) Receive(topicName string, subscriptionName string) (msg Message, err error) {
	var topic *Topic
	topic, err = b.GetTopic(topicName)
	if err != nil {
		return
	}

	sub := topic.GetSubscription(subscriptionName)
	if sub == nil {
		err = fmt.Errorf("subscription not found")
		return
	}

	return b.readNextMessage(topic, sub.(*subscription))
}

func (b *broker) readNextMessage(topic *Topic, sub *subscription) (Message, error) {
	partition := topic.GetPartition(sub.Cursor().PartitionID)

	for {
		if err := b.initializeCursorIfNeeded(partition, sub); err != nil {
			time.Sleep(100 * time.Millisecond)
			continue
		}

		ledger, err := b.bkClient.OpenLedger(sub.cursor.LedgerID)
		if err != nil {
			return Message{}, fmt.Errorf("failed to open ledger: %v", err)
		}

		msg, found, err := b.tryReadNextEntry(ledger, sub)
		if err != nil {
			return Message{}, err
		}
		if found {
			return msg, nil
		}

		if b.moveToNextLedgerIfAvailable(partition, sub) {
			continue
		}

		time.Sleep(100 * time.Millisecond)
	}
}

func (b *broker) initializeCursorIfNeeded(partition *Partition, sub *subscription) error {
	if sub.cursor.LedgerID == 0 {
		if len(partition.Ledgers) == 0 {
			return fmt.Errorf("no ledgers available")
		}
		sub.cursor.LedgerID = partition.Ledgers[0]
		sub.cursor.EntryID = 0
	}
	return nil
}

func (b *broker) tryReadNextEntry(ledger Ledger, sub *subscription) (Message, bool, error) {
	lastConfirmed := ledger.GetLastAddConfirmed()
	nextEntryID := sub.cursor.EntryID + 1

	if nextEntryID <= lastConfirmed {
		payload, err := ledger.ReadEntry(nextEntryID)
		if err != nil {
			return Message{}, false, fmt.Errorf("failed to read entry: %v", err)
		}

		msg := Message{
			ID: MessageID{
				PartitionID: sub.cursor.PartitionID,
				LedgerID:    sub.cursor.LedgerID,
				EntryID:     nextEntryID,
			},
			Content: payload,
		}

		sub.ackMessages[msg.ID] = true
		sub.cursor.EntryID = nextEntryID

		return msg, true, nil
	}

	return Message{}, false, nil
}

func (b *broker) moveToNextLedgerIfAvailable(partition *Partition, sub *subscription) bool {
	ledgerIndex := partition.IndexOfLedger(sub.cursor.LedgerID)
	if ledgerIndex+1 < len(partition.Ledgers) {
		sub.cursor.LedgerID = partition.Ledgers[ledgerIndex+1]
		sub.cursor.EntryID = 0
		return true
	}
	return false
}

func (b *broker) routeMessage(topic *Topic, msg Message) (partition *Partition, ledger Ledger, err error) {
	// 先分区
	partition = topic.GetPartition(b.selectPartition(topic, msg))

	// 再分ledger，ledger 是 pulsar 比 Kafka 更灵活的最根本设计
	ledger, err = b.getOrCreateLedger(partition)
	return
}

func (b *broker) processDelayedMessages() {
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
				log.Printf("%s delay message is due: %v", b.logIdent(), msg)
				if err := b.Publish(msg); err != nil {
					log.Printf("Error publishing delayed message: %v", err)
				}
			}
		}
	}
}

func (b *broker) manageLedgerRetention() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			b.cleanupExpiredLedgers()
		}
	}
}

func (b *broker) cleanupExpiredLedgers() {
	b.mu.Lock()
	defer b.mu.Unlock()

	retentionMaxAge := 48 * time.Hour
	cutoffTime := time.Now().Add(retentionMaxAge)
	log.Printf("%s Starting ledger retention cleanup, cutoff time: %v", b.logIdent(), cutoffTime)

	for _, topic := range b.topics {
		for _, partition := range topic.Partitions {
			var activeLedgers []LedgerID
			for _, ledgerID := range partition.Ledgers {
				ledger, err := b.bkClient.OpenLedger(ledgerID)
				if err != nil {
					log.Printf("%s Failed to open ledger %d for retention check: %v", b.logIdent(), ledgerID, err)
					continue
				}

				if ledger.Age() > retentionMaxAge {
					ledger.Close()
					b.bkClient.DeleteLedger(ledgerID)
				} else {
					activeLedgers = append(activeLedgers, ledgerID)
				}
			}

			partition.Ledgers = activeLedgers
		}
	}
}

func (b *broker) selectPartition(topic *Topic, msg Message) PartitionID {
	// 实际上类似 Kafka Partitioner，可能根据消息key分区
	return PartitionID(0)
}

func (b *broker) getOrCreateLedger(partition *Partition) (Ledger, error) {
	var ledger Ledger
	var err error

	if len(partition.Ledgers) == 0 {
		// Initialize
		ledger, err = b.createNewLedger(partition)
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
			ledger, err = b.createNewLedger(partition)
			if err != nil {
				return nil, err
			}
		}
	}

	return ledger, nil
}

func (b *broker) createNewLedger(partition *Partition) (Ledger, error) {
	ledger, err := b.bkClient.CreateLedger(LedgerOption{
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

func (b *broker) shouldRolloverLedger(ledger Ledger) bool {
	// e,g. based on the number of entries or age of the ledger
	return false
}

func (b *broker) rebalance() {
}

func (b *broker) logIdent() string {
	return "Broker[" + b.info.ID + "]"
}
