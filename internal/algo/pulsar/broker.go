package main

import "fmt"

type Broker interface {
	CreateTopic(name string) (*Topic, error)
	GetTopic(name string) (*Topic, error)
	DeleteTopic(name string) error

	CreateProducer(topic string) (Producer, error)
	Subscribe(topic, subscriptionName string, subType SubscriptionType) (Consumer, error)
}

// BrokerError 定义了 Broker 操作可能返回的错误
type BrokerError struct {
	Op  string
	Err error
}

func (e *BrokerError) Error() string {
	return fmt.Sprintf("broker %s error: %v", e.Op, e.Err)
}
