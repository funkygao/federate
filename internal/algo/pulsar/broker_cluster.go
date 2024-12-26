package main

import (
	"sync"
)

type BrokerCluster interface {
}

type InMemoryBrokerCluster struct {
	brokers []Broker
	mu      sync.RWMutex

	zookeeper ZooKeeper
}

func (bc InMemoryBrokerCluster) selectBroker() Broker {
	return nil
}
