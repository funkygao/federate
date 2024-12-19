package table

import (
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
	// In a real implementation, this would reorganize data based on a clustering strategy
	// For this example, we'll just create a new instant
	instant := Instant{
		Timestamp: time.Now(),
		Action:    Clustering,
		State:     "COMPLETED",
	}
	table.Timeline.AddInstant(instant)
	return nil
}

type Clusterer struct {
	table *Table
}

func NewClusterer(table *Table) *Clusterer {
	return &Clusterer{table: table}
}

func (c *Clusterer) Cluster() error {
	strategy := c.table.ClusteringStrategy
	if strategy == nil {
		strategy = &SimpleClusteringStrategy{MaxFileCount: 2}
	}

	if strategy.ShouldCluster(c.table) {
		err := strategy.PerformClustering(c.table)
		if err != nil {
			return err
		}

		instant := Instant{
			Timestamp: time.Now(),
			Action:    Clustering,
			State:     "COMPLETED",
		}
		c.table.Timeline.AddInstant(instant)
	}

	return nil
}
