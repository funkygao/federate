package main

import "fmt"

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
	Close() error
}

// ConsumerError 定义了 Consumer 操作可能返回的错误
type ConsumerError struct {
	Op  string
	Err error
}

func (e *ConsumerError) Error() string {
	return fmt.Sprintf("consumer %s error: %v", e.Op, e.Err)
}
