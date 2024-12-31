package main

type SnapshotIsolation interface {
	BeginSnapshot(timestamp TimeInterval) Snapshot
}

type snapshotIsolation struct {
	mvcc MVCC
}

func NewSnapshotIsolation(mvcc MVCC) SnapshotIsolation {
	return &snapshotIsolation{mvcc: mvcc}
}

func (si *snapshotIsolation) BeginSnapshot(timestamp TimeInterval) Snapshot {
	return &snapshot{timestamp: timestamp, mvcc: si.mvcc}
}

type Snapshot interface {
	Read(key string) (interface{}, error)
}

type snapshot struct {
	timestamp TimeInterval
	mvcc      MVCC
}

func (s *snapshot) Read(key string) (interface{}, error) {
	// We use the latest possible time in the interval for reads
	return s.mvcc.Read(key, s.timestamp.Latest)
}
