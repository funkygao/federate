package main

type ZooKeeper interface {
	RegisterBroker(BrokerInfo) error
	RegisterTopic(Topic) error
	RegisterLedger(LedgerInfo) error
}

type BrokerInfo struct {
	ID   string
	Host string
	Port int
}

type LedgerInfo struct {
	LedgerID     LedgerID
	LedgerOption LedgerOption
	Bookies      []Bookie
}

func getZooKeeper() ZooKeeper {
	return &zooKeeper{}
}

type zooKeeper struct {
}

func (zk *zooKeeper) RegisterBroker(BrokerInfo) error {
	return nil
}

func (zk *zooKeeper) RegisterTopic(Topic) error {
	return nil
}

func (zk *zooKeeper) RegisterLedger(LedgerInfo) error {
	return nil
}
