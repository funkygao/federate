package main

// ZooKeeper interface defines the operations Pulsar uses to interact with ZooKeeper.
// Pulsar uses ZooKeeper for storing metadata and coordinating between brokers.
//
// The ZooKeeper znode structure for Pulsar typically looks like this:
//
//	/
//	├── admin/
//	│   ├── clusters/
//	│   └── policies/
//	├── brokers/
//	│   └── <broker-id>/
//	├── managed-ledgers/
//	│   └── <namespace>/
//	│       └── <topic>/
//	│           └── persistent/
//	│               └── <ledger-id>
//	├── loadbalance/
//	│   └── brokers/
//	│       └── <broker-id>
//	├── namespace/
//	├── schemas/
//	└── topics/
//	    └── persistent/
//	        └── <tenant>/
//	            └── <namespace>/
//	                └── <topic>/
//	                    ├── partitions/
//	                    ├── compaction
//	                    └── subscriptions/
//	                        └── <subscription-name>/
//	                            ├── position
//	                            └── state
//
// This structure allows Pulsar to store and manage metadata for brokers, topics,
// subscriptions, and other components in a hierarchical manner.
type ZooKeeper interface {
	RegisterBroker(BrokerInfo) error
	RegisterTopic(Topic) error
	RegisterLedger(LedgerInfo) error
	DeleteLedger(LedgerID)

	NextLedgerID() LedgerID
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

func (zk *zooKeeper) DeleteLedger(LedgerID) {
}

func (zk *zooKeeper) NextLedgerID() LedgerID {
	return 0
}
