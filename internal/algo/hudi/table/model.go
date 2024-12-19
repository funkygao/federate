package table

import "time"

type Record struct {
	Key       string         `parquet:"name=key, type=BYTE_ARRAY, convertedtype=UTF8, encoding=PLAIN_DICTIONARY"`
	Fields    map[string]any `parquet:"name=fields, type=MAP, convertedtype=MAP, keytype=BYTE_ARRAY, keyconvertedtype=UTF8"`
	Timestamp time.Time      `parquet:"name=timestamp, type=INT96"`
}

type TableType int

const (
	CopyOnWrite TableType = iota
	MergeOnRead
)

type CommitType string

const (
	Insert    CommitType = "INSERT"
	Update    CommitType = "UPDATE"
	Delete    CommitType = "DELETE"
	Compacted CommitType = "COMPACTED"
	Clustered CommitType = "CLUSTERED"
)

type Commit struct {
	Timestamp time.Time
	Type      CommitType
	Records   []Record
}
