package main

import (
	"fmt"
	"sync"
)

type TopicManager struct {
	topics map[string]*Topic
	mu     sync.RWMutex
}

func NewTopicManager() *TopicManager {
	return &TopicManager{
		topics: make(map[string]*Topic),
	}
}

func (tm *TopicManager) CreateTopic(name string) (*Topic, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.topics[name]; exists {
		return nil, fmt.Errorf("topic already exists")
	}

	topic := &Topic{
		Name:       name,
		Partitions: make(map[PartitionID]*Partition),
	}
	tm.topics[name] = topic
	return topic, nil
}

func (tm *TopicManager) GetTopic(name string) (*Topic, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	topic, exists := tm.topics[name]
	if !exists {
		return nil, fmt.Errorf("topic not found")
	}
	return topic, nil
}

func (tm *TopicManager) DeleteTopic(name string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.topics[name]; !exists {
		return fmt.Errorf("topic not found")
	}

	delete(tm.topics, name)
	return nil
}

func (tm *TopicManager) AddPartitionToTopic(topicName string, partitionID PartitionID) error {
	topic, err := tm.GetTopic(topicName)
	if err != nil {
		return err
	}

	topic.Partitions[partitionID] = &Partition{
		ID:           partitionID,
		TimeSegments: make(map[TimeSegmentID]*TimeSegment),
	}
	return nil
}

func (tm *TopicManager) GetPartitionFromTopic(topicName string, partitionID PartitionID) (*Partition, error) {
	topic, err := tm.GetTopic(topicName)
	if err != nil {
		return nil, err
	}

	partition, exists := topic.Partitions[partitionID]
	if !exists {
		return nil, fmt.Errorf("partition not found")
	}
	return partition, nil
}
