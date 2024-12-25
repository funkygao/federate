package main

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
}

type InMemoryConsumer struct {
	subscription Subscription
}

func NewInMemoryConsumer(sub Subscription) *InMemoryConsumer {
	return &InMemoryConsumer{
		subscription: sub,
	}
}

func (c *InMemoryConsumer) Receive() (Message, error) {
	msg, err := c.subscription.Receive()
	if err != nil {
		return Message{}, err
	}
	return msg, nil
}

func (c *InMemoryConsumer) Acknowledge(msgID MessageID) error {
	return c.subscription.Acknowledge(msgID)
}
