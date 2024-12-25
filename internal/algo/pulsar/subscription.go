package main

import "fmt"

type SubscriptionType int

const (
	Exclusive SubscriptionType = iota
	Shared
	Failover
)

type Subscription interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
	Unsubscribe() error
}

type SubscriptionError struct {
	Op  string
	Err error
}

func (e *SubscriptionError) Error() string {
	return fmt.Sprintf("subscription %s error: %v", e.Op, e.Err)
}
