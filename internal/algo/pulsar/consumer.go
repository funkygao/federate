package main

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
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
		return Message{}, err
	}
	return msg, nil
}

func (c *InMemoryConsumer) Acknowledge(msgID MessageID) error {
	err := c.subscription.Acknowledge(msgID)
	if err != nil {
		return err
	}
	return nil
}
