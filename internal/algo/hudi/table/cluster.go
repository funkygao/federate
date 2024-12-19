package table

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"
)

type ClusteringStrategy interface {
	ShouldCluster(table *Table) bool
	PerformClustering(table *Table) error
}

type SimpleClusteringStrategy struct {
	MaxFileCount int
}

func (s *SimpleClusteringStrategy) ShouldCluster(table *Table) bool {
	files, err := table.FS.List(table.Name)
	if err != nil {
		return false
	}
	return len(files) > s.MaxFileCount
}

func (s *SimpleClusteringStrategy) PerformClustering(table *Table) error {
	records, err := table.Read()
	if err != nil {
		return err
	}

	// Simple clustering: just rewrite all records into a single file
	clusteredFilePath := filepath.Join(table.Name, fmt.Sprintf("clustered_%d.json", time.Now().UnixNano()))
	data, err := json.Marshal(records)
	if err != nil {
		return err
	}

	if err := table.FS.Write(clusteredFilePath, data); err != nil {
		return err
	}

	// Delete old files
	files, err := table.FS.List(table.Name)
	if err != nil {
		return err
	}

	for _, file := range files {
		if file != filepath.Base(clusteredFilePath) {
			if err := table.FS.Delete(filepath.Join(table.Name, file)); err != nil {
				return err
			}
		}
	}

	commit := Commit{
		Timestamp: time.Now(),
		Type:      Clustered,
	}
	table.Timeline.AddCommit(commit)

	return nil
}

type Clusterer struct {
	table *Table
}

func NewClusterer(table *Table) *Clusterer {
	return &Clusterer{table: table}
}

func (c *Clusterer) Cluster() error {
	// In a real implementation, this would reorganize data based on a clustering strategy
	// For this example, we'll just create a new commit
	commit := Commit{
		Timestamp: time.Now(),
		Type:      Clustered,
	}
	c.table.Timeline.AddCommit(commit)
	return nil
}
