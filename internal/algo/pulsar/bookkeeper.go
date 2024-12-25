package main

import (
	"fmt"
	"sync"
)

type BookKeeper interface {
	AddEntry(partitionID PartitionID, segmentID TimeSegmentID, ledgerID LedgerID, entry Payload) (EntryID, error)
	ReadEntry(partitionID PartitionID, segmentID TimeSegmentID, ledgerID LedgerID, entryID EntryID) (Payload, error)
	CreateLedger(partitionID PartitionID, segmentID TimeSegmentID) (LedgerID, error)
}

type InMemoryBookKeeper struct {
	partitions map[PartitionID]Partition
	mu         sync.RWMutex
}

func NewInMemoryBookKeeper() *InMemoryBookKeeper {
	return &InMemoryBookKeeper{
		partitions: make(map[PartitionID]Partition),
	}
}

func (bk *InMemoryBookKeeper) AddEntry(partitionID PartitionID, segmentID TimeSegmentID, ledgerID LedgerID, entry Payload) (EntryID, error) {
	bk.mu.Lock()
	defer bk.mu.Unlock()

	partition, ok := bk.partitions[partitionID]
	if !ok {
		return 0, fmt.Errorf("partition not found")
	}

	for i, segment := range partition.TimeSegments {
		if segment.ID == segmentID {
			for j, ledger := range segment.Ledgers {
				if ledger.ID == ledgerID {
					entryID := EntryID(len(ledger.Entries))
					ledger.Entries[entryID] = entry
					partition.TimeSegments[i].Ledgers[j] = ledger
					bk.partitions[partitionID] = partition
					return entryID, nil
				}
			}
			return 0, fmt.Errorf("ledger not found")
		}
	}
	return 0, fmt.Errorf("segment not found")
}

func (bk *InMemoryBookKeeper) ReadEntry(partitionID PartitionID, segmentID TimeSegmentID, ledgerID LedgerID, entryID EntryID) (Payload, error) {
	bk.mu.RLock()
	defer bk.mu.RUnlock()

	partition, ok := bk.partitions[partitionID]
	if !ok {
		return nil, fmt.Errorf("partition not found")
	}

	for _, segment := range partition.TimeSegments {
		if segment.ID == segmentID {
			for _, ledger := range segment.Ledgers {
				if ledger.ID == ledgerID {
					entry, ok := ledger.Entries[entryID]
					if !ok {
						return nil, fmt.Errorf("entry not found")
					}
					return entry, nil
				}
			}
			return nil, fmt.Errorf("ledger not found")
		}
	}
	return nil, fmt.Errorf("segment not found")
}

func (bk *InMemoryBookKeeper) CreateLedger(partitionID PartitionID, segmentID TimeSegmentID) (LedgerID, error) {
	bk.mu.Lock()
	defer bk.mu.Unlock()

	partition, ok := bk.partitions[partitionID]
	if !ok {
		partition = Partition{ID: partitionID, TimeSegments: []TimeSegment{}}
		bk.partitions[partitionID] = partition
	}

	var segment *TimeSegment
	for i, seg := range partition.TimeSegments {
		if seg.ID == segmentID {
			segment = &partition.TimeSegments[i]
			break
		}
	}
	if segment == nil {
		segment = &TimeSegment{ID: segmentID, Ledgers: []Ledger{}}
		partition.TimeSegments = append(partition.TimeSegments, *segment)
	}

	ledgerID := LedgerID(len(segment.Ledgers))
	newLedger := Ledger{ID: ledgerID, Entries: make(map[EntryID]Payload)}
	segment.Ledgers = append(segment.Ledgers, newLedger)

	bk.partitions[partitionID] = partition
	return ledgerID, nil
}
