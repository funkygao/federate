package main

import (
	"time"
)

type Client interface {
	CreateProducer(ProducerOptions) (Producer, error)
	Subscribe(ConsumerOptions) (Consumer, error)
}

type ProducerOptions struct {
	Topic                   string
	Name                    string
	SendTimeout             time.Duration
	MaxPendingMessages      int
	HashingScheme           HashingScheme
	CompressionType         CompressionType
	BatchingMaxPublishDelay time.Duration
	BatchingMaxMessages     uint
}

type ConsumerOptions struct {
	Topic               string
	SubscriptionName    string
	Type                SubscriptionType
	ReceiverQueueSize   int
	NackRedeliveryDelay time.Duration
}

// HashingScheme 定义消息路由到分区的哈希方案
// 类似 Kafka Partitioner
type HashingScheme int

type CompressionType int
