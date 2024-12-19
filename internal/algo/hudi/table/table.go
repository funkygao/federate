package table

import (
	"fmt"
	"time"

	"federate/internal/algo/hudi/meta"
	"federate/internal/algo/hudi/store"
)

type Table struct {
	Name               string
	Type               TableType
	FS                 store.FileSystem
	Index              Index
	Timeline           *Timeline
	Metadata           *meta.Metadata
	Schema             *Schema
	ClusteringStrategy ClusteringStrategy
	FileGroups         map[string]FileGroup // Key is partition path
	LatestCommitTime   time.Time
}

func NewTable(name string, tableType TableType, fs store.FileSystem, schema *Schema) *Table {
	return &Table{
		Name:               name,
		Type:               tableType,
		FS:                 fs,
		Timeline:           NewTimeline(),
		Metadata:           meta.NewMetadata(),
		Schema:             schema,
		ClusteringStrategy: &SimpleClusteringStrategy{MaxFileCount: 2},
		FileGroups:         make(map[string]FileGroup),
		Index:              NewBloomFilterIndex(),
	}
}

func (t *Table) Upsert(records []Record) error {
	for _, record := range records {
		if err := t.Schema.Validate(record); err != nil {
			return fmt.Errorf("record validation failed: %w", err)
		}
	}

	switch t.Type {
	case MergeOnRead:
		return t.upsertMergeOnRead(records)
	case CopyOnWrite:
		return t.upsertCopyOnWrite(records)
	default:
		return fmt.Errorf("unsupported table type")
	}
}

func (t *Table) Read() ([]Record, error) {
	switch t.Type {
	case MergeOnRead:
		return t.readMergeOnRead()
	case CopyOnWrite:
		return t.readCopyOnWrite()
	default:
		return nil, fmt.Errorf("unsupported table type")
	}
}

func (t *Table) Delete(keys []string) error {
	switch t.Type {
	case CopyOnWrite:
		return t.deleteCopyOnWrite(keys)
	case MergeOnRead:
		return t.deleteMergeOnRead(keys)
	default:
		return fmt.Errorf("unsupported table type: %v", t.Type)
	}
}

func (t *Table) updateTimeline(action InstantType, records []Record) {
	instant := Instant{
		Timestamp: time.Now(),
		Action:    action,
		State:     "COMPLETED",
	}
	t.Timeline.AddInstant(instant)
	t.LatestCommitTime = instant.Timestamp
}
