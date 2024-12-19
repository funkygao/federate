package main

import (
	"time"
)

type MVCCReader struct {
	table *Table
}

func NewMVCCReader(table *Table) *MVCCReader {
	return &MVCCReader{table: table}
}

func (r *MVCCReader) ReadAsOf(timestamp time.Time) ([]Record, error) {
	commits := r.table.Timeline.GetCommitsAfter(timestamp)
	var records []Record

	for _, commit := range commits {
		for _, record := range commit.Records {
			if record.Timestamp.Before(timestamp) || record.Timestamp.Equal(timestamp) {
				records = append(records, record)
			}
		}
	}

	return records, nil
}
