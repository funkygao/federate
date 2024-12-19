package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"path/filepath"
	"time"
)

func main() {
	fs, err := NewLocalFileSystem("./data")
	if err != nil {
		log.Fatalf("Failed to initialize file system: %v", err)
	}

	// Define schema
	schema := &Schema{
		Fields: []SchemaField{
			{Name: "name", Type: StringType},
			{Name: "age", Type: IntType},
			{Name: "active", Type: BoolType},
			{Name: "last_login", Type: DateTimeType},
		},
	}

	// Create table
	table := NewTable("users", MergeOnRead, fs, schema)

	// Read initial records from JSON file
	data, err := ioutil.ReadFile("records.json")
	if err != nil {
		log.Fatalf("Failed to read records file: %v", err)
	}

	var records []Record
	if err := json.Unmarshal(data, &records); err != nil {
		log.Fatalf("Failed to unmarshal records: %v", err)
	}

	// Upsert records into the table
	err = table.Upsert(records)
	if err != nil {
		log.Fatalf("Failed to insert records: %v", err)
	}
	log.Println("Inserted initial records")

	// Read records
	readRecords, err := table.ReadMergeOnRead()
	if err != nil {
		log.Fatalf("Failed to read records: %v", err)
	}
	log.Println("Read records:")
	for _, record := range readRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Update a record
	updateRecords := []Record{
		{
			Key: "1",
			Fields: map[string]any{
				"name":       "Alice Smith",
				"age":        31,
				"active":     true,
				"last_login": time.Now(),
			},
			Timestamp: time.Now(),
		},
	}
	err = table.Upsert(updateRecords)
	if err != nil {
		log.Fatalf("Failed to update record: %v", err)
	}
	log.Println("Updated record")

	// Compact small files
	compactor := NewCompactor(table)
	err = compactor.MergeSmallFiles(1024) // 1KB threshold
	if err != nil {
		log.Fatalf("Failed to compact small files: %v", err)
	}
	log.Println("Compacted small files")

	// Perform clustering
	if table.ClusteringStrategy.ShouldCluster(table) {
		err = table.ClusteringStrategy.PerformClustering(table)
		if err != nil {
			log.Fatalf("Failed to perform clustering: %v", err)
		}
		log.Println("Performed clustering")
	}

	// Write to Parquet
	parquetRecords, err := table.ReadMergeOnRead()
	if err != nil {
		log.Fatalf("Failed to read records for Parquet: %v", err)
	}
	err = table.WriteParquet(parquetRecords, filepath.Join("./data", table.Name, "users.parquet"))
	if err != nil {
		log.Fatalf("Failed to write Parquet file: %v", err)
	}
	log.Println("Wrote records to Parquet file")

	// Read from Parquet
	parquetReadRecords, err := table.ReadParquet(filepath.Join("./data", table.Name, "users.parquet"))
	if err != nil {
		log.Fatalf("Failed to read Parquet file: %v", err)
	}
	log.Println("Read records from Parquet file:")
	for _, record := range parquetReadRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Use MVCC reader
	mvccReader := NewMVCCReader(table)
	mvccRecords, err := mvccReader.ReadAsOf(time.Now())
	if err != nil {
		log.Fatalf("Failed to read records with MVCC: %v", err)
	}
	log.Println("Read records with MVCC:")
	for _, record := range mvccRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Print timeline
	log.Println("Timeline:")
	for _, commit := range table.Timeline.GetCommitsAfter(time.Time{}) {
		log.Printf("Commit Type: %s, Timestamp: %v", commit.Type, commit.Timestamp)
	}

	// Print index
	index, err := table.Index.GetAll()
	if err != nil {
		log.Fatalf("Failed to get index: %v", err)
	}
	log.Println("Index:")
	for key, filePath := range index {
		log.Printf("Key: %s, File Path: %s", key, filePath)
	}

	// Set and get metadata
	table.Metadata.Set("last_compaction", time.Now().Format(time.RFC3339))
	lastCompaction, _ := table.Metadata.Get("last_compaction")
	log.Printf("Last compaction: %s", lastCompaction)
}

func init() {
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)
}
