package main

type Producer interface {
	Send(msg Message) error
}

type InMemoryProducer struct {
	broker *InMemoryBroker
	topic  *Topic
}

func NewInMemoryProducer(broker *InMemoryBroker, topic *Topic) *InMemoryProducer {
	return &InMemoryProducer{
		broker: broker,
		topic:  topic,
	}
}

func (p *InMemoryProducer) Send(msg Message) error {
	msg.Topic = p.topic.Name
	return p.broker.publishMessage(p.topic.Name, msg)
}
