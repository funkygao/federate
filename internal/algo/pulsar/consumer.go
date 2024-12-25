package main

type Consumer interface {
	Receive() (Message, error)
	Acknowledge(msgID MessageID) error
}
