package main

type Consumer interface {
	Fetch() (Message, error)
	Ack(msgID MessageID) error
}

type InMemoryConsumer struct {
	subscription Subscription
}

func NewInMemoryConsumer(sub Subscription) *InMemoryConsumer {
	return &InMemoryConsumer{
		subscription: sub,
	}
}

func (c *InMemoryConsumer) Fetch() (Message, error) {
	msg, err := c.subscription.Fetch()
	if err != nil {
		return Message{}, err
	}
	return msg, nil
}

func (c *InMemoryConsumer) Ack(msgID MessageID) error {
	return c.subscription.Ack(msgID)
}
