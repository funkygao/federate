package main

type ZooKeeper interface {
	RegisterBroker(brokerID string, info BrokerInfo) error
	GetBrokers() (map[string]BrokerInfo, error)

	RegisterLedger(ledgerID LedgerID, metadata LedgerMetadata) error
	GetLedgerMetadata(ledgerID LedgerID) (LedgerMetadata, error)
}

type InMemoryZooKeeper struct {
}
