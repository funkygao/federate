package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"federate/internal/algo/hudi/store"
	"federate/internal/algo/hudi/table"
)

func main() {
	fs, err := store.NewLocalFileSystem("./data")
	if err != nil {
		log.Fatalf("Failed to initialize file system: %v", err)
	}

	// Define the schema
	schema := &table.Schema{
		Fields: []table.SchemaField{
			{Name: "name", Type: table.StringType},
			{Name: "age", Type: table.IntType},
			{Name: "active", Type: table.BoolType},
			{Name: "last_login", Type: table.DateTimeType},
		},
	}

	// Read initial records from JSON file
	data, err := ioutil.ReadFile("records.json")
	if err != nil {
		log.Fatalf("Failed to read records file: %v", err)
	}

	var initialRecords []table.Record
	if err := json.Unmarshal(data, &initialRecords); err != nil {
		log.Fatalf("Failed to unmarshal records: %v", err)
	}

	// Run demos for both table types
	if err := demoCopyOnWrite(fs, schema, initialRecords); err != nil {
		log.Printf("%v", err)
	}
	if err := demoMergeOnRead(fs, schema, initialRecords); err != nil {
		log.Printf("%v", err)
	}

	log.Println("Demos completed")
}

func demoCopyOnWrite(fs store.FileSystem, schema *table.Schema, initialRecords []table.Record) error {
	log.Println("=== Copy-on-Write Table Demo ===")

	// Create a Copy-on-Write table
	cowTable := table.NewTable("users_cow", table.CopyOnWrite, fs, schema)

	// Upsert initial records into the Copy-on-Write table
	if err := cowTable.Upsert(initialRecords); err != nil {
		return fmt.Errorf("Failed to upsert records into Copy-on-Write table: %v", err)
	}
	log.Println("Upserted initial records into Copy-on-Write table")

	// Read records from the Copy-on-Write table
	cowRecords, err := cowTable.Read()
	if err != nil {
		return fmt.Errorf("Failed to read records from Copy-on-Write table: %v", err)
	}
	log.Println("Records in Copy-on-Write table:")
	for _, record := range cowRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Update a record in the Copy-on-Write table
	updateRecordsCoW := []table.Record{
		{
			Key: "1",
			Fields: map[string]interface{}{
				"name":       "Alice Smith",
				"age":        31,
				"active":     true,
				"last_login": time.Now().Format(time.RFC3339),
			},
			Timestamp: time.Now(),
		},
	}
	if err := cowTable.Upsert(updateRecordsCoW); err != nil {
		return fmt.Errorf("Failed to update records in Copy-on-Write table: %v", err)
	}
	log.Println("Updated records in Copy-on-Write table")

	// Read records again from the Copy-on-Write table
	cowRecords, err = cowTable.Read()
	if err != nil {
		return fmt.Errorf("Failed to read records after update from Copy-on-Write table: %v", err)
	}
	log.Println("Records in Copy-on-Write table after update:")
	for _, record := range cowRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	log.Println("=== End of Copy-on-Write Table Demo ===")
	return nil
}

func demoMergeOnRead(fs store.FileSystem, schema *table.Schema, initialRecords []table.Record) error {
	log.Println("=== Merge-on-Read Table Demo ===")

	// Create a Merge-on-Read table
	morTable := table.NewTable("users_mor", table.MergeOnRead, fs, schema)

	// Upsert initial records into the Merge-on-Read table
	if err := morTable.Upsert(initialRecords); err != nil {
		return fmt.Errorf("Failed to upsert records into Merge-on-Read table: %v", err)
	}
	log.Println("Upserted initial records into Merge-on-Read table")

	// Read records from the Merge-on-Read table
	morRecords, err := morTable.Read()
	if err != nil {
		return fmt.Errorf("Failed to read records from Merge-on-Read table: %v", err)
	}
	log.Println("Records in Merge-on-Read table:")
	for _, record := range morRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Update a record in the Merge-on-Read table
	updateRecordsMoR := []table.Record{
		{
			Key: "2",
			Fields: map[string]interface{}{
				"name":       "Bob Johnson",
				"age":        26,
				"active":     false,
				"last_login": time.Now().Format(time.RFC3339),
			},
			Timestamp: time.Now(),
		},
	}
	if err := morTable.Upsert(updateRecordsMoR); err != nil {
		return fmt.Errorf("Failed to update records in Merge-on-Read table: %v", err)
	}
	log.Println("Updated records in Merge-on-Read table")

	// Read records again from the Merge-on-Read table
	morRecords, err = morTable.Read()
	if err != nil {
		return fmt.Errorf("Failed to read records after update from Merge-on-Read table: %v", err)
	}
	log.Println("Records in Merge-on-Read table after update:")
	for _, record := range morRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	// Perform compaction on the Merge-on-Read table
	compactor := table.NewCompactor(morTable)
	if err := compactor.MergeSmallFiles(1024); err != nil {
		return fmt.Errorf("Failed to compact small files in Merge-on-Read table: %v", err)
	}
	log.Println("Compacted small files in Merge-on-Read table")

	// Read records again after compaction
	morRecords, err = morTable.Read()
	if err != nil {
		return fmt.Errorf("Failed to read records after compaction from Merge-on-Read table: %v", err)
	}
	log.Println("Records in Merge-on-Read table after compaction:")
	for _, record := range morRecords {
		log.Printf("Key: %s, Fields: %v", record.Key, record.Fields)
	}

	log.Println("=== End of Merge-on-Read Table Demo ===")
	return nil
}

func init() {
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)
}
