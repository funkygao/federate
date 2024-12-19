package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type Table struct {
	Name               string
	Type               TableType
	FS                 FileSystem
	Index              *FSIndex
	Timeline           *Timeline
	Metadata           *Metadata
	Schema             *Schema
	ClusteringStrategy ClusteringStrategy
}

func NewTable(name string, tableType TableType, fs FileSystem, schema *Schema) *Table {
	table := &Table{
		Name:               name,
		Type:               tableType,
		FS:                 fs,
		Timeline:           NewTimeline(),
		Metadata:           NewMetadata(),
		Schema:             schema,
		ClusteringStrategy: &SimpleClusteringStrategy{MaxFileCount: 2},
	}
	table.Index = NewFSIndex(table)
	return table
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

func (t *Table) upsertCopyOnWrite(records []Record) error {
	// Step 1: Validate records
	for _, record := range records {
		if err := t.Schema.Validate(record); err != nil {
			return fmt.Errorf("record validation failed: %w", err)
		}
	}

	// Step 2: Read existing data
	existingRecords, err := t.readAllDataParquet()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read existing data: %w", err)
	}

	// Step 3: Merge records
	mergedRecords := t.mergeRecords(existingRecords, records)

	// Step 4: Write merged data
	dataFilePath := filepath.Join(t.Name, "data.parquet")
	if err := t.WriteParquet(mergedRecords, dataFilePath); err != nil {
		return fmt.Errorf("failed to write merged data: %w", err)
	}

	// Step 5: Update index
	t.Index = NewFSIndex(t)
	for _, record := range mergedRecords {
		t.Index.Add(record.Key, dataFilePath)
	}

	// Step 6: Update timeline
	commit := Commit{
		Timestamp: time.Now(),
		Type:      Update,
		Records:   records,
	}
	t.Timeline.AddCommit(commit)

	return nil
}

func (t *Table) readAllDataParquet() ([]Record, error) {
	dataFilePath := filepath.Join(t.Name, "data.parquet")

	// Check if the data file exists
	if _, err := t.FS.Read(dataFilePath); err != nil {
		return nil, err
	}

	// Read records from Parquet file
	records, err := t.ReadParquet(dataFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read Parquet file: %w", err)
	}
	return records, nil
}

func (t *Table) mergeRecords(existingRecords []Record, newRecords []Record) []Record {
	recordMap := make(map[string]Record)

	// Add existing records to the map
	for _, record := range existingRecords {
		recordMap[record.Key] = record
	}

	// Merge new records
	for _, record := range newRecords {
		existingRecord, exists := recordMap[record.Key]
		if !exists || record.Timestamp.After(existingRecord.Timestamp) {
			recordMap[record.Key] = record
		}
	}

	// Convert map back to slice
	mergedRecords := make([]Record, 0, len(recordMap))
	for _, record := range recordMap {
		mergedRecords = append(mergedRecords, record)
	}

	return mergedRecords
}

func (t *Table) upsertMergeOnRead(records []Record) error {
	if t.Type != MergeOnRead {
		return fmt.Errorf("table type is not MergeOnRead")
	}

	deltaFilePath := filepath.Join(t.Name, fmt.Sprintf("delta_%d.json", time.Now().UnixNano()))
	data, err := json.Marshal(records)
	if err != nil {
		return err
	}

	if err := t.FS.Write(deltaFilePath, data); err != nil {
		return err
	}

	commit := Commit{
		Timestamp: time.Now(),
		Type:      Update,
		Records:   records,
	}
	t.Timeline.AddCommit(commit)

	return nil
}

func (t *Table) ReadMergeOnRead() ([]Record, error) {
	if t.Type != MergeOnRead {
		return nil, fmt.Errorf("table type is not MergeOnRead")
	}

	files, err := t.FS.List(t.Name)
	if err != nil {
		return nil, err
	}

	var allRecords []Record
	for _, file := range files {
		if filepath.Ext(file) == ".json" {
			data, err := t.FS.Read(filepath.Join(t.Name, file))
			if err != nil {
				return nil, err
			}

			var records []Record
			if err := json.Unmarshal(data, &records); err != nil {
				return nil, err
			}

			allRecords = append(allRecords, records...)
		}
	}

	// Merge records based on the latest timestamp
	mergedRecords := make(map[string]Record)
	for _, record := range allRecords {
		if existing, ok := mergedRecords[record.Key]; !ok || record.Timestamp.After(existing.Timestamp) {
			mergedRecords[record.Key] = record
		}
	}

	var result []Record
	for _, record := range mergedRecords {
		result = append(result, record)
	}

	return result, nil
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

func (t *Table) deleteCopyOnWrite(keys []string) error {
	// Step 1: Read existing data
	existingRecords, err := t.readAllDataParquet()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read existing data: %w", err)
	}

	// Step 2: Remove records matching the keys
	keySet := make(map[string]struct{})
	for _, key := range keys {
		keySet[key] = struct{}{}
	}

	updatedRecords := make([]Record, 0)
	for _, record := range existingRecords {
		if _, exists := keySet[record.Key]; !exists {
			updatedRecords = append(updatedRecords, record)
		}
	}

	// Step 3: Write updated data
	dataFilePath := filepath.Join(t.Name, "data.parquet")
	if err := t.WriteParquet(updatedRecords, dataFilePath); err != nil {
		return fmt.Errorf("failed to write updated data: %w", err)
	}

	// Step 4: Update index
	t.Index = NewFSIndex(t)
	for _, record := range updatedRecords {
		t.Index.Add(record.Key, dataFilePath)
	}

	// Step 5: Update timeline
	commit := Commit{
		Timestamp: time.Now(),
		Type:      Delete,
		Records:   nil, // Optionally include deleted records
	}
	t.Timeline.AddCommit(commit)

	return nil
}

func (t *Table) deleteMergeOnRead(keys []string) error {
	return fmt.Errorf("Not implemented")
}
