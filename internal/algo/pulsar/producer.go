package main

import "time"

type Producer interface {
	Send(msg Message) error
}

type producer struct {
	broker Broker
	topic  *Topic
}

func NewProducer(broker Broker, topic *Topic) *producer {
	return &producer{broker, topic}
}

func (p *producer) Send(msg Message) error {
	msg.Topic = p.topic.Name
	msg.Timestamp = time.Now() // 客户端时间

	return p.broker.Publish(msg)
}
