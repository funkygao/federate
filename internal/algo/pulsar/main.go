package main

import (
	"fmt"
	"log"
	"time"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)

	// 创建一个有 3 个 Bookie 的 BookKeeper 服务
	bk := NewInMemoryBookKeeperService(3)
	broker := NewPulsarBroker(bk)

	topic := "test-topic"

	go produce(broker, topic)
	go consume(broker, topic)

	// 等待足够的时间让所有消息被发布和消费
	time.Sleep(20 * time.Second)
}

func produce(broker Broker, topic string) {
	for i := 0; i < 10; i++ {
		delay := time.Duration(0)
		if i%3 == 0 {
			delay = 5 * time.Second
		}
		msg := Message{
			Content:   Payload(fmt.Sprintf("Message %d", i)),
			Timestamp: time.Now(),
			Delay:     delay,
		}
		err := broker.Publish(topic, msg)
		if err != nil {
			log.Printf("Failed to publish message: %v", err)
		}
		log.Printf("Published [P:%d-S:%d-L:%d-E:%d] %s, Delay: %v",
			msg.ID.PartitionID, msg.ID.TimeSegmentID, msg.ID.LedgerID, msg.ID.EntryID,
			msg.Content, msg.Delay)
		time.Sleep(time.Second)
	}
}

func consume(broker Broker, topic string) {
	subName := "test-subscription"
	sub, err := broker.Subscribe(topic, subName, Shared)
	if err != nil {
		log.Fatalf("Failed to subscribe: %v", err)
	}

	consumer := NewPulsarConsumer(sub)
	for {
		msg, err := consumer.Receive()
		if err != nil {
			log.Printf("Failed to receive message: %v", err)
			continue
		}
		log.Printf("Received  [P:%d-S:%d-L:%d-E:%d] %s at %v",
			msg.ID.PartitionID, msg.ID.TimeSegmentID, msg.ID.LedgerID, msg.ID.EntryID,
			msg.Content, time.Now().Format("15:04:05.000000"))
		err = consumer.Acknowledge(msg.ID)
		if err != nil {
			log.Printf("Failed to acknowledge message: %v", err)
		}
	}
}
