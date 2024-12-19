package table

import (
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"time"
)

func (t *Table) upsertMergeOnRead(records []Record) error {
	deltaFilePath := filepath.Join(t.Name, fmt.Sprintf("delta_%d.json", time.Now().UnixNano()))
	data, err := json.Marshal(records)
	if err != nil {
		return err
	}

	log.Printf("delta file: %s", deltaFilePath)

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

func (t *Table) readMergeOnRead() ([]Record, error) {
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

func (t *Table) deleteMergeOnRead(keys []string) error {
	return fmt.Errorf("Not implemented")
}
