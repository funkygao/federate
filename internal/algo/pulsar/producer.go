package main

type Producer interface {
	Send(msg Message) error
}
