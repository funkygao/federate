package main

import (
	"encoding/json"
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

	// Demonstrate Copy-on-Write table
	log.Println("=== Copy-on-Write Table Demo ===")
	cowTable := table.NewTable("users_cow", table.CopyOnWrite, fs, schema)
	if err := demoCopyOnWrite(cowTable, initialRecords); err != nil {
		log.Printf("%v", err)
	}

	// Demonstrate Merge-on-Read table
	log.Println("\n=== Merge-on-Read Table Demo ===")
	morTable := table.NewTable("users_mor", table.MergeOnRead, fs, schema)
	if err := demoMergeOnRead(morTable, initialRecords); err != nil {
		log.Printf("%v", err)
	}
}

func demoCopyOnWrite(t *table.Table, initialRecords []table.Record) error {
	// Insert initial records
	if err := t.Upsert(initialRecords); err != nil {
		return err
	}
	log.Println("Inserted initial records")

	// Read records
	records, err := t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records", len(records))

	// Update a record
	updateRecord := table.Record{
		Key: "1",
		Fields: map[string]interface{}{
			"name":       "Alice Smith",
			"age":        31,
			"active":     true,
			"last_login": time.Now().Format(time.RFC3339),
		},
		Timestamp: time.Now(),
	}
	if err := t.Upsert([]table.Record{updateRecord}); err != nil {
		return err
	}
	log.Println("Updated record")

	// Read records again
	records, err = t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records after update", len(records))

	// Delete a record
	if err := t.Delete([]string{"2"}); err != nil {
		return err
	}
	log.Println("Deleted record")

	// Read records one last time
	records, err = t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records after delete", len(records))
	return nil
}

func demoMergeOnRead(t *table.Table, initialRecords []table.Record) error {
	// Insert initial records
	if err := t.Upsert(initialRecords); err != nil {
		return err
	}
	log.Println("Inserted initial records")

	// Read records
	records, err := t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records", len(records))

	// Update a record
	updateRecord := table.Record{
		Key: "1",
		Fields: map[string]interface{}{
			"name":       "Alice Johnson",
			"age":        32,
			"active":     false,
			"last_login": time.Now().Format(time.RFC3339),
		},
		Timestamp: time.Now(),
	}
	if err := t.Upsert([]table.Record{updateRecord}); err != nil {
		return err
	}
	log.Println("Updated record")

	// Read records again
	records, err = t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records after update", len(records))

	// Perform compaction
	compactor := table.NewCompactor(t)
	if err := compactor.Compact(); err != nil {
		return err
	}
	log.Println("Performed compaction")

	// Read records after compaction
	records, err = t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records after compaction", len(records))

	// Delete a record
	if err := t.Delete([]string{"2"}); err != nil {
		return err
	}
	log.Println("Deleted record")

	// Read records one last time
	records, err = t.Read()
	if err != nil {
		return err
	}
	log.Printf("Read %d records after delete\n", len(records))
	return nil
}

func init() {
	log.SetFlags(log.Lshortfile | log.Lmicroseconds)
}
