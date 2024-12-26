package main

import (
	"fmt"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)

	broker := initializeBroker()
	broker.Start()

	topic := createTopic(broker, "test-topic")

	producer := createProducer(broker, topic)
	consumer := createConsumer(broker, topic)

	go produceMessages(producer, 20)
	go consumeMessages(consumer)

	time.Sleep(30 * time.Second)
	log.Println("Shutting down...")
}

func initializeBroker() Broker {
	bk := NewInMemoryBookKeeper(3)
	broker := NewInMemoryBroker(bk)
	return broker
}

func createTopic(broker Broker, topicName string) *Topic {
	topic, err := broker.CreateTopic(topicName)
	if err != nil {
		log.Fatalf("Failed to create topic: %v", err)
	}

	log.Printf("Created topic: %s", topicName)
	return topic
}

func createProducer(broker Broker, topic *Topic) Producer {
	producer, err := broker.CreateProducer(topic.Name)
	if err != nil {
		log.Fatalf("Failed to create producer: %v", err)
	}

	log.Printf("Created producer for topic: %s", topic.Name)
	return producer
}

func createConsumer(broker Broker, topic *Topic) Consumer {
	consumer, err := broker.CreateConsumer(topic.Name, "test-subscription", Shared)
	if err != nil {
		log.Fatalf("Failed to create consumer: %v", err)
	}

	log.Printf("Created consumer for topic: %s", topic.Name)
	return consumer
}

func produceMessages(producer Producer, N int) {
	for i := 0; i < N; i++ {
		msg := Message{
			Content:   []byte(fmt.Sprintf("Message %d", i)),
			Timestamp: time.Now(),
		}
		if i%5 == 0 {
			msg.Delay = 2 * time.Second
		}

		if err := producer.Send(msg); err != nil {
			log.Printf("Failed to send message: %v", err)
		} else {
			log.Printf("Sent message: %s", string(msg.Content))
		}

		time.Sleep(500 * time.Millisecond)
	}
}

func consumeMessages(consumer Consumer) {
	for {
		msg, err := consumer.Fetch()
		if err != nil {
			log.Printf("Failed to receive message: %v", err)
			time.Sleep(time.Second)
			continue
		}

		log.Printf("Received message: %s", string(msg.Content))
		if err = consumer.Ack(msg.ID); err != nil {
			log.Printf("Failed to ack message: %v", err)
		}
	}
}
