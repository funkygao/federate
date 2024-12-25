package main

type SubscriptionType int

const (
	Exclusive SubscriptionType = iota
	Shared
	Failover
)

type Subscription interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
	AddConsumer(consumer Consumer) error
	RemoveConsumer(consumer Consumer) error
}
