package main

import (
	"fmt"
	"time"
)

func main() {
	// Initialize components
	mvcc := NewMVCC()
	truetime := NewTrueTime(10 * time.Millisecond)

	// Create regions and replicas
	region1 := NewRegion("region1")
	region2 := NewRegion("region2")

	replica1 := NewReplica("replica1", mvcc)
	replica2 := NewReplica("replica2", mvcc)
	replica3 := NewReplica("replica3", mvcc)

	region1.AddReplica(replica1)
	region1.AddReplica(replica2)
	region2.AddReplica(replica3)

	regions := []Region{region1, region2}

	// Create Spanner instance
	spanner := NewSpanner(regions, truetime, mvcc)

	// Perform some operations
	fmt.Println("Writing initial data...")
	spanner.Write("key1", "value1")
	spanner.Write("key2", "value2")

	fmt.Println("Reading data...")
	value1, _ := spanner.Read("key1")
	value2, _ := spanner.Read("key2")
	fmt.Printf("key1: %v, key2: %v\n", value1, value2)

	fmt.Println("Performing a transaction...")
	tx := spanner.BeginTransaction()
	tx.Write("key3", "value3")
	tx.Write("key4", "value4")
	err := tx.Commit()
	if err != nil {
		fmt.Printf("Transaction failed: %v\n", err)
	} else {
		fmt.Println("Transaction committed successfully")
	}

	fmt.Println("Reading data after transaction...")
	value3, _ := spanner.Read("key3")
	value4, _ := spanner.Read("key4")
	fmt.Printf("key3: %v, key4: %v\n", value3, value4)

	// Demonstrate snapshot isolation
	fmt.Println("Demonstrating snapshot isolation...")
	snapshotIsolation := NewSnapshotIsolation(mvcc)
	snapshot := snapshotIsolation.BeginSnapshot(truetime.Now())

	// Write new data
	spanner.Write("key1", "new_value1")

	// Read from snapshot (should still see old value)
	oldValue1, _ := snapshot.Read("key1")
	newValue1, _ := spanner.Read("key1")
	fmt.Printf("Snapshot read of key1: %v, Current value of key1: %v\n", oldValue1, newValue1)

	// Demonstrate TrueTime
	fmt.Println("Demonstrating TrueTime...")
	interval := truetime.Now()
	fmt.Printf("Current time interval: Earliest %v, Latest %v\n", interval.Earliest, interval.Latest)
	fmt.Printf("Interval duration: %v\n", interval.Latest.Sub(interval.Earliest))
}
