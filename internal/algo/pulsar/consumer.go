package main

import "fmt"

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
	Close() error
}

// ConsumerError 定义了 Consumer 操作可能返回的错误
type ConsumerError struct {
	Op  string
	Err error
}

func (e *ConsumerError) Error() string {
	return fmt.Sprintf("consumer %s error: %v", e.Op, e.Err)
}

type InMemoryConsumer struct {
	subscription *InMemorySubscription
}

func NewInMemoryConsumer(sub *InMemorySubscription) *InMemoryConsumer {
	return &InMemoryConsumer{
		subscription: sub,
	}
}

func (c *InMemoryConsumer) Receive() (Message, error) {
	msg, err := c.subscription.Receive()
	if err != nil {
		return Message{}, &ConsumerError{Op: "Receive", Err: err}
	}
	return msg, nil
}

func (c *InMemoryConsumer) Acknowledge(msgID MessageID) error {
	err := c.subscription.Acknowledge(msgID)
	if err != nil {
		return &ConsumerError{Op: "Acknowledge", Err: err}
	}
	return nil
}

func (c *InMemoryConsumer) Close() error {
	// 实际实现中，这里可能需要清理资源或通知 Broker
	return nil
}
