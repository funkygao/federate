package main

type TailCache interface {
	Put(Topic, Message)
	GetLastMessage(Topic) (Message, bool)
}
