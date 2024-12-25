package main

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
}

type PulsarConsumer struct {
	subscription Subscription
}

func NewPulsarConsumer(sub Subscription) *PulsarConsumer {
	consumer := &PulsarConsumer{subscription: sub}
	sub.AddConsumer(consumer)
	return consumer
}

func (c *PulsarConsumer) Receive() (Message, error) {
	return c.subscription.Receive()
}

func (c *PulsarConsumer) Acknowledge(msgID MessageID) error {
	return c.subscription.Acknowledge(msgID)
}
