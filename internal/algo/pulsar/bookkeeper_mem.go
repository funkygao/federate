package main

import (
	"math/rand"
	"sync"
)

type InMemoryBookKeeperService struct {
	bookies []Bookie

	partitions map[PartitionID]Partition
	mu         sync.RWMutex

	nextLedgerID LedgerID
}

func NewInMemoryBookKeeperService(bookieCount int) BookKeeperService {
	bookies := make([]Bookie, bookieCount)
	for i := 0; i < bookieCount; i++ {
		bookies[i] = NewInMemoryBookie()
	}
	return &InMemoryBookKeeperService{
		bookies:      bookies,
		partitions:   make(map[PartitionID]Partition),
		nextLedgerID: 0,
	}
}

func (bks *InMemoryBookKeeperService) AddEntry(partitionID PartitionID, timeSegmentID TimeSegmentID, ledgerID LedgerID, entry Payload) (EntryID, error) {
	bks.mu.Lock()
	defer bks.mu.Unlock()

	// 选择多个 Bookie 进行复制（这里简化为选择 3 个或所有 Bookie，取较小值）
	replicationFactor := 3
	if len(bks.bookies) < replicationFactor {
		replicationFactor = len(bks.bookies)
	}

	selectedBookies := make([]Bookie, replicationFactor)
	for i := 0; i < replicationFactor; i++ {
		selectedBookies[i] = bks.bookies[(int(ledgerID)+i)%len(bks.bookies)]
	}

	// 生成新的 EntryID
	entryID := EntryID(0) // 在实际实现中，这应该是基于 ledger 的递增值

	// 将 entry 添加到选中的所有 Bookie
	for _, bookie := range selectedBookies {
		err := bookie.AddEntry(ledgerID, entryID, entry)
		if err != nil {
			return 0, err
		}
	}

	// 更新元数据（简化实现）
	partition, ok := bks.partitions[partitionID]
	if !ok {
		partition = Partition{ID: partitionID, TimeSegments: []TimeSegment{}}
		bks.partitions[partitionID] = partition
	}

	var timeSegment *TimeSegment
	for i, ts := range partition.TimeSegments {
		if ts.ID == timeSegmentID {
			timeSegment = &partition.TimeSegments[i]
			break
		}
	}
	if timeSegment == nil {
		timeSegment = &TimeSegment{ID: timeSegmentID, Ledgers: []Ledger{}}
		partition.TimeSegments = append(partition.TimeSegments, *timeSegment)
	}

	var ledger *Ledger
	for i, l := range timeSegment.Ledgers {
		if l.ID == ledgerID {
			ledger = &timeSegment.Ledgers[i]
			break
		}
	}
	if ledger == nil {
		ledger = &Ledger{ID: ledgerID, Entries: map[EntryID]Payload{}}
		timeSegment.Ledgers = append(timeSegment.Ledgers, *ledger)
	}

	ledger.Entries[entryID] = entry
	bks.partitions[partitionID] = partition

	return entryID, nil
}

func (bks *InMemoryBookKeeperService) ReadEntry(partitionID PartitionID, timeSegmentID TimeSegmentID, ledgerID LedgerID, entryID EntryID) (Payload, error) {
	bks.mu.RLock()
	defer bks.mu.RUnlock()

	// 随机选择一个 Bookie 来读取数据
	bookieIndex := rand.Intn(len(bks.bookies))
	bookie := bks.bookies[bookieIndex]

	return bookie.ReadEntry(ledgerID, entryID)
}

func (bks *InMemoryBookKeeperService) CreateLedger(partitionID PartitionID, timeSegmentID TimeSegmentID) (LedgerID, error) {
	bks.mu.Lock()
	defer bks.mu.Unlock()

	ledgerID := bks.nextLedgerID
	bks.nextLedgerID++

	partition, ok := bks.partitions[partitionID]
	if !ok {
		partition = Partition{ID: partitionID, TimeSegments: []TimeSegment{}}
		bks.partitions[partitionID] = partition
	}

	var timeSegment *TimeSegment
	for i, ts := range partition.TimeSegments {
		if ts.ID == timeSegmentID {
			timeSegment = &partition.TimeSegments[i]
			break
		}
	}
	if timeSegment == nil {
		timeSegment = &TimeSegment{ID: timeSegmentID, Ledgers: []Ledger{}}
		partition.TimeSegments = append(partition.TimeSegments, *timeSegment)
	}

	newLedger := Ledger{ID: ledgerID, Entries: make(map[EntryID]Payload)}
	timeSegment.Ledgers = append(timeSegment.Ledgers, newLedger)

	bks.partitions[partitionID] = partition

	return ledgerID, nil
}
