package main

import (
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)

	broker := NewBroker(NewBookKeeper(3))
	broker.Start()

	topic, _ := broker.CreateTopic("OrderCreated")

	producer, _ := broker.CreateProducer(topic.Name)
	consumer, _ := broker.CreateConsumer(topic.Name, "test-subscription", Shared)

	go produceMessages(producer, 12, 3)

	// Delay starting the consumer
	time.Sleep(1 * time.Second)

	go consumeMessages(consumer)

	time.Sleep(20 * time.Second)
	log.Println("Shutting down...")
}

func produceMessages(producer Producer, N, delayGap int) {
	for i := 0; i < N; i++ {
		msg := Message{Content: []byte(fmt.Sprintf("Order[%d]", i))}
		if i%delayGap == 0 {
			msg.Delay = 2 * time.Second
		}

		if err := producer.Send(msg); err != nil {
			log.Printf("[%d] Failed to send message: %v", i, err)
		} else {
			log.Printf("[%d] Sent message: %s", i, msg)
		}

		time.Sleep(1200 * time.Millisecond)
	}
}

func consumeMessages(consumer Consumer) {
	for {
		msg, err := consumer.Receive()
		if err != nil {
			log.Printf("Failed to receive message: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}

		log.Printf("Received message: %s", msg)

		if err = consumer.Ack(msg.ID); err != nil {
			log.Printf("Failed to ack message: %v", err)
		}
	}
}
