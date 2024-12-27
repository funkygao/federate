package main

type Consumer interface {
	Receive() (Message, error)

	// Kafka ack offet, while pulsar ack (LedgerID, EntryID)
	Ack(msgID MessageID) error
}

type consumer struct {
	subscription Subscription
}

func NewConsumer(sub Subscription) *consumer {
	return &consumer{sub}
}

func (c *consumer) Receive() (Message, error) {
	msg, err := c.subscription.Receive()
	if err != nil {
		return Message{}, err
	}
	return msg, nil
}

func (c *consumer) Ack(msgID MessageID) error {
	return c.subscription.Ack(msgID)
}
