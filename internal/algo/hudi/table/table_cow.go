package table

import (
	"fmt"
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
	dataFilePath := filepath.Join(t.Name, fmt.Sprintf("data_%d.parquet", time.Now().UnixNano()))
	if err := t.WriteParquet(mergedRecords, dataFilePath); err != nil {
		return fmt.Errorf("failed to write merged data: %w", err)
	}

	// Update index
	for _, record := range mergedRecords {
		t.Index.Add(record.Key, FileLocation{
			PartitionPath:  t.Name,
			FileSliceIndex: 0,
			IsBaseFile:     true,
			FilePath:       dataFilePath,
		})
	}

	// Update timeline
	instant := Instant{
		Timestamp: time.Now(),
		Action:    Commit,
		State:     "COMPLETED",
	}
	t.Timeline.AddInstant(instant)

	return nil
}

func (t *Table) readCopyOnWrite() ([]Record, error) {
	files, err := t.FS.List(t.Name)
	if err != nil {
		return nil, err
	}

	var allRecords []Record
	for _, file := range files {
		if filepath.Ext(file) == ".parquet" {
			records, err := t.ReadParquet(filepath.Join(t.Name, file))
			if err != nil {
				return nil, err
			}
			allRecords = append(allRecords, records...)
		}
	}

	return allRecords, nil
}

func (t *Table) mergeRecords(existingRecords, newRecords []Record) []Record {
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
	// Read existing data
	existingRecords, err := t.readCopyOnWrite()
	if err != nil {
		return fmt.Errorf("failed to read existing data: %w", err)
	}

	// Remove records with matching keys
	updatedRecords := make([]Record, 0)
	for _, record := range existingRecords {
		shouldDelete := false
		for _, key := range keys {
			if record.Key == key {
				shouldDelete = true
				break
			}
		}
		if !shouldDelete {
			updatedRecords = append(updatedRecords, record)
		}
	}

	// Write updated data
	dataFilePath := filepath.Join(t.Name, fmt.Sprintf("data_%d.parquet", time.Now().UnixNano()))
	if err := t.WriteParquet(updatedRecords, dataFilePath); err != nil {
		return fmt.Errorf("failed to write updated data: %w", err)
	}

	// Update index
	for _, key := range keys {
		t.Index.Remove(key)
	}
	for _, record := range updatedRecords {
		t.Index.Add(record.Key, FileLocation{
			PartitionPath:  t.Name,
			FileSliceIndex: 0,
			IsBaseFile:     true,
			FilePath:       dataFilePath,
		})
	}

	// Update timeline
	instant := Instant{
		Timestamp: time.Now(),
		Action:    Commit,
		State:     "COMPLETED",
	}
	t.Timeline.AddInstant(instant)

	return nil
}
