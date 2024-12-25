package main

import (
	"fmt"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)

	// 初始化系统组件
	broker := initializeBroker()

	// 创建主题
	topic := createTopic(broker, "test-topic")

	// 创建生产者和消费者
	producer := createProducer(broker, topic)
	consumer := createConsumer(broker, topic)

	// 启动消息生产和消费
	go produceMessages(producer)
	go consumeMessages(consumer)

	// 运行一段时间后退出
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
	consumer, err := broker.Subscribe(topic.Name, "test-subscription", Shared)
	if err != nil {
		log.Fatalf("Failed to create consumer: %v", err)
	}
	log.Printf("Created consumer for topic: %s", topic.Name)
	return consumer
}

func produceMessages(producer Producer) {
	for i := 0; i < 20; i++ {
		msg := Message{
			Content:   []byte(fmt.Sprintf("Message %d", i)),
			Timestamp: time.Now(),
		}
		if i%5 == 0 {
			msg.Delay = 2 * time.Second
		}
		err := producer.Send(msg)
		if err != nil {
			log.Printf("Failed to send message: %v", err)
		} else {
			log.Printf("Sent message: %s", string(msg.Content))
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func consumeMessages(consumer Consumer) {
	for {
		msg, err := consumer.Receive()
		if err != nil {
			log.Printf("Failed to receive message: %v", err)
			time.Sleep(time.Second) // 添加短暂的睡眠以避免过于频繁的日志输出
			continue
		}
		log.Printf("Received message: %s", string(msg.Content))
		err = consumer.Acknowledge(msg.ID)
		if err != nil {
			log.Printf("Failed to acknowledge message: %v", err)
		}
	}
}
