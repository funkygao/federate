package main

type Producer interface {
	Send(msg Message) error
}

type InMemoryProducer struct {
	broker Broker
	topic  *Topic
}

func NewInMemoryProducer(broker Broker, topic *Topic) *InMemoryProducer {
	return &InMemoryProducer{
		broker: broker,
		topic:  topic,
	}
}

func (p *InMemoryProducer) Send(msg Message) error {
	msg.Topic = p.topic.Name
	return p.broker.Publish(p.topic.Name, msg)
}
