package main

var (
	_ PartitionLB = (*InMemoryBroker)(nil)
	_ BrokerLB    = (*InMemoryBrokerCluster)(nil)
)

type BrokerLB interface {
	selectBroker() Broker
}

type PartitionLB interface {
	selectPartition(msg Message) PartitionID
}
