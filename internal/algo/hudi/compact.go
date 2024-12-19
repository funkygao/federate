package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"
)

type Compactor struct {
	table *Table
}

func NewCompactor(table *Table) *Compactor {
	return &Compactor{table: table}
}

func (c *Compactor) Compact() error {
	// In a real implementation, this would merge small files and optimize storage
	// For this example, we'll just create a new commit
	commit := Commit{
		Timestamp: time.Now(),
		Type:      Compacted,
	}
	c.table.Timeline.AddCommit(commit)
	return nil
}

func (c *Compactor) MergeSmallFiles(sizeThreshold int64) error {
	files, err := c.table.FS.List(c.table.Name)
	if err != nil {
		return err
	}

	var smallFiles []string
	var allRecords []Record

	for _, file := range files {
		fullPath := filepath.Join(c.table.Name, file)
		data, err := c.table.FS.Read(fullPath)
		if err != nil {
			return err
		}

		if int64(len(data)) < sizeThreshold {
			smallFiles = append(smallFiles, fullPath)

			var records []Record
			if err := json.Unmarshal(data, &records); err != nil {
				return err
			}
			allRecords = append(allRecords, records...)
		}
	}

	if len(smallFiles) > 1 {
		mergedFilePath := filepath.Join(c.table.Name, fmt.Sprintf("merged_%d.json", time.Now().UnixNano()))
		mergedData, err := json.Marshal(allRecords)
		if err != nil {
			return err
		}

		if err := c.table.FS.Write(mergedFilePath, mergedData); err != nil {
			return err
		}

		// Delete small files
		for _, file := range smallFiles {
			if err := c.table.FS.Delete(file); err != nil {
				return err
			}
		}

		commit := Commit{
			Timestamp: time.Now(),
			Type:      Compacted,
		}
		c.table.Timeline.AddCommit(commit)
	}

	return nil
}
