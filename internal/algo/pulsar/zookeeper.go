package main

type ZooKeeper interface {
	RegisterBroker(BrokerInfo) error
	RegisterLedger(LedgerMetadata) error
}

type BrokerInfo struct {
	ID   string
	Host string
	Port int
}

type LedgerMetadata struct {
	ID LedgerID
}

func getZooKeeper() ZooKeeper {
	return &InMemoryZooKeeper{}
}

type InMemoryZooKeeper struct {
}

func (zk *InMemoryZooKeeper) RegisterBroker(BrokerInfo) error {
	return nil
}

func (zk *InMemoryZooKeeper) RegisterLedger(LedgerMetadata) error {
	return nil
}
