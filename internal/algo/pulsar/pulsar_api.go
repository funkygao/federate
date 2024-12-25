package main

import (
	"context"
	"time"
)

type Client interface {
	CreateProducer(ProducerOptions) (Producer, error)
	Subscribe(ConsumerOptions) (Consumer, error)
	CreateReader(ReaderOptions) (Reader, error)
	TopicPartitions(topic string) ([]string, error)
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

// Reader 表示消息读取器，用于从特定位置读取消息
type Reader interface {
	// Next 读取下一条消息
	Next(context.Context) (Message, error)

	// HasNext 检查是否还有更多消息
	HasNext() bool
}

type ReaderOptions struct {
	Topic          string
	StartMessageID MessageID
	ReadCompacted  bool
}

type ConsumerOptions struct {
	Topic               string
	SubscriptionName    string
	Type                SubscriptionType
	ReceiverQueueSize   int
	NackRedeliveryDelay time.Duration
}

// HashingScheme 定义消息路由到分区的哈希方案
type HashingScheme int

type CompressionType int

// NewClient 创建一个新的 Pulsar 客户端
func NewClient(opts ClientOptions) (Client, error) {
	return nil, nil
}

type ClientOptions struct {
	URL                        string
	OperationTimeout           time.Duration
	ConnectionTimeout          time.Duration
	TLSAllowInsecureConnection bool
	TLSTrustCertsFilePath      string
}
