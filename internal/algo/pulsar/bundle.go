package main

import "fmt"

// Bundle represents a group of topics
type Bundle interface {
	ID() string
	AddTopic(*Topic) error
	RemoveTopic(string) error
	GetTopic(string) (*Topic, error)
}

type SimpleBundle struct {
	id     string
	topics map[string]*Topic
}

func NewSimpleBundle(id string) *SimpleBundle {
	return &SimpleBundle{
		id:     id,
		topics: make(map[string]*Topic),
	}
}

func (b *SimpleBundle) ID() string {
	return b.id
}

func (b *SimpleBundle) AddTopic(topic *Topic) error {
	if _, exists := b.topics[topic.Name]; exists {
		return fmt.Errorf("topic %s already exists in bundle", topic.Name)
	}
	b.topics[topic.Name] = topic
	return nil
}

func (b *SimpleBundle) RemoveTopic(topicName string) error {
	if _, exists := b.topics[topicName]; !exists {
		return fmt.Errorf("topic %s does not exist in bundle", topicName)
	}
	delete(b.topics, topicName)
	return nil
}

func (b *SimpleBundle) GetTopic(topicName string) (*Topic, error) {
	topic, exists := b.topics[topicName]
	if !exists {
		return nil, fmt.Errorf("topic %s not found in bundle", topicName)
	}
	return topic, nil
}
