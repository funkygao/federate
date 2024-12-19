package table

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"
)

func (t *Table) upsertMergeOnRead(records []Record) error {
	// Group records by partition
	partitionedRecords := t.partitionRecords(records)

	for partitionPath, partitionRecords := range partitionedRecords {
		fileGroup, exists := t.FileGroups[partitionPath]
		if !exists {
			fileGroup = FileGroup{PartitionPath: partitionPath}
			t.FileGroups[partitionPath] = fileGroup
		}

		if len(fileGroup.FileSlices) == 0 || t.shouldCreateNewFileSlice(fileGroup.FileSlices[len(fileGroup.FileSlices)-1]) {
			// Create a new file slice
			baseFilePath := t.createBaseFile(partitionPath, partitionRecords)
			fileGroup.FileSlices = append(fileGroup.FileSlices, FileSlice{BaseFile: baseFilePath})
		} else {
			// Append to existing file slice
			lastSlice := &fileGroup.FileSlices[len(fileGroup.FileSlices)-1]
			deltaFilePath := t.createDeltaFile(partitionPath, partitionRecords)
			lastSlice.DeltaFiles = append(lastSlice.DeltaFiles, deltaFilePath)
		}

		t.FileGroups[partitionPath] = fileGroup

		// Update index
		for _, record := range partitionRecords {
			t.Index.Add(record.Key, FileLocation{
				PartitionPath:  partitionPath,
				FileSliceIndex: len(fileGroup.FileSlices) - 1,
				IsBaseFile:     false,
				FilePath:       fileGroup.FileSlices[len(fileGroup.FileSlices)-1].DeltaFiles[len(fileGroup.FileSlices[len(fileGroup.FileSlices)-1].DeltaFiles)-1],
			})
		}
	}

	// Update timeline
	instant := Instant{
		Timestamp: time.Now(),
		Action:    DeltaCommit,
		State:     "COMPLETED",
	}
	t.Timeline.AddInstant(instant)

	return nil
}

func (t *Table) readMergeOnRead() ([]Record, error) {
	var allRecords []Record

	for _, fileGroup := range t.FileGroups {
		for _, slice := range fileGroup.FileSlices {
			// Read base file
			baseRecords, err := t.ReadParquet(slice.BaseFile)
			if err != nil {
				return nil, fmt.Errorf("failed to read base file %s: %w", slice.BaseFile, err)
			}
			allRecords = append(allRecords, baseRecords...)

			// Read and apply delta files
			for _, deltaFile := range slice.DeltaFiles {
				deltaRecords, err := t.readDeltaFile(deltaFile)
				if err != nil {
					return nil, fmt.Errorf("failed to read delta file %s: %w", deltaFile, err)
				}
				allRecords = t.applyDelta(allRecords, deltaRecords)
			}
		}
	}

	return allRecords, nil
}

func (t *Table) deleteMergeOnRead(keys []string) error {
	deleteRecords := make([]Record, len(keys))
	for i, key := range keys {
		deleteRecords[i] = Record{
			Key:       key,
			Timestamp: time.Now(),
			Fields:    map[string]interface{}{"_deleted": true},
		}
	}

	return t.upsertMergeOnRead(deleteRecords)
}

func (t *Table) partitionRecords(records []Record) map[string][]Record {
	// In a real implementation, this would partition records based on a partitioning scheme
	// For simplicity, we'll just put all records in a single partition
	return map[string][]Record{"default": records}
}

func (t *Table) shouldCreateNewFileSlice(slice FileSlice) bool {
	// In a real implementation, this would decide based on file sizes, record counts, etc.
	return len(slice.DeltaFiles) >= 5
}

func (t *Table) createBaseFile(partitionPath string, records []Record) string {
	baseFilePath := filepath.Join(partitionPath, fmt.Sprintf("base_%d.parquet", time.Now().UnixNano()))
	t.WriteParquet(records, baseFilePath)
	return baseFilePath
}

func (t *Table) createDeltaFile(partitionPath string, records []Record) string {
	deltaFilePath := filepath.Join(partitionPath, fmt.Sprintf("delta_%d.json", time.Now().UnixNano()))
	data, _ := json.Marshal(records)
	t.FS.Write(deltaFilePath, data)
	return deltaFilePath
}

func (t *Table) readDeltaFile(path string) ([]Record, error) {
	data, err := t.FS.Read(path)
	if err != nil {
		return nil, err
	}

	var records []Record
	err = json.Unmarshal(data, &records)
	return records, err
}

func (t *Table) applyDelta(baseRecords, deltaRecords []Record) []Record {
	recordMap := make(map[string]Record)

	for _, record := range baseRecords {
		recordMap[record.Key] = record
	}

	for _, record := range deltaRecords {
		if deleted, ok := record.Fields["_deleted"].(bool); ok && deleted {
			delete(recordMap, record.Key)
		} else {
			recordMap[record.Key] = record
		}
	}

	result := make([]Record, 0, len(recordMap))
	for _, record := range recordMap {
		result = append(result, record)
	}

	return result
}
