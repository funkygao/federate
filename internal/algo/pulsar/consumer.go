package main

type Consumer interface {
	Receive() (Message, error)
	Ack(MessageID) error
}

type consumer struct {
	broker       Broker
	topic        *Topic
	subscription Subscription
}

func NewConsumer(b Broker, t *Topic, sub Subscription) *consumer {
	return &consumer{b, t, sub}
}

func (c *consumer) Receive() (Message, error) {
	msg, err := c.broker.Receive(c.topic.Name, c.subscription.Name())
	if err != nil {
		return Message{}, err
	}
	return msg, nil
}

func (c *consumer) Ack(msgID MessageID) error {
	return c.broker.Ack(c.topic.Name, c.subscription.Name(), msgID)
}
