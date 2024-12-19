package table

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

func (t *Table) upsertCopyOnWrite(records []Record) error {
	// Read existing data
	existingRecords, err := t.readCopyOnWrite()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read existing data: %w", err)
	}

	// Merge records
	mergedRecords := t.mergeRecords(existingRecords, records)

	// Write merged data
	dataFilePath := filepath.Join(t.Name, "data.parquet")
	if err := t.WriteParquet(mergedRecords, dataFilePath); err != nil {
		return fmt.Errorf("failed to write merged data: %w", err)
	}

	// Update index
	for _, record := range mergedRecords {
		t.Index.Add(record.Key, dataFilePath)
	}

	// Update timeline
	commit := Commit{
		Timestamp: time.Now(),
		Type:      Update,
		Records:   records,
	}
	t.Timeline.AddCommit(commit)

	return nil
}

func (t *Table) readCopyOnWrite() ([]Record, error) {
	dataFilePath := filepath.Join(t.Name, "data.parquet")

	log.Printf("data parquet: %s", dataFilePath)

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

func (t *Table) deleteCopyOnWrite(keys []string) error {
	// Step 1: Read existing data
	existingRecords, err := t.readCopyOnWrite()
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
