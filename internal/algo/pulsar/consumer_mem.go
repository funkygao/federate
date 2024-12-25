package main

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
