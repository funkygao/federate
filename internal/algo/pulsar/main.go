package main

import (
	"fmt"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)

	bk := NewInMemoryBookKeeper()
	broker := NewPulsarBroker(bk)

	topic := "test-topic"
	subName := "test-subscription"
	sub, err := broker.Subscribe(topic, subName, Shared)
	if err != nil {
		log.Fatalf("Failed to subscribe: %v", err)
	}

	consumer := NewPulsarConsumer(sub)

	// 发布消息，包括延迟消息
	go func() {
		for i := 0; i < 10; i++ {
			delay := time.Duration(0)
			if i%3 == 0 {
				delay = 5 * time.Second
			}
			msg := Message{
				Content:   fmt.Sprintf("Message %d", i),
				Timestamp: time.Now(),
				Delay:     delay,
			}
			err := broker.Publish(topic, msg)
			if err != nil {
				log.Printf("Failed to publish message: %v", err)
			}
			log.Printf("Published [ID:%d-%d] %s, Delay: %v", msg.ID.LedgerID, msg.ID.EntryID, msg.Content, msg.Delay)
			time.Sleep(time.Second)
		}
	}()

	// 消费消息
	go func() {
		for i := 0; i < 10; i++ {
			msg, err := consumer.Receive()
			if err != nil {
				log.Printf("Failed to receive message: %v", err)
				continue
			}
			log.Printf("Received  [ID:%d-%d] %s at %v", msg.ID.LedgerID, msg.ID.EntryID, msg.Content, time.Now().Format("15:04:05.000000"))
			err = consumer.Acknowledge(msg.ID)
			if err != nil {
				log.Printf("Failed to acknowledge message: %v", err)
			}
		}
	}()

	// 等待足够的时间让所有消息被发布和消费
	time.Sleep(20 * time.Second)
}
