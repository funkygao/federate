package main

import (
	"sync"
	"time"
)

type MVCC interface {
	Read(key string, timestamp time.Time) (any, error)
	Write(key string, value any, timestamp time.Time) error
}

type mvcc struct {
	data map[string][]versionedValue
	mu   sync.RWMutex
}

type versionedValue struct {
	value     any
	timestamp time.Time
}

func NewMVCC() MVCC {
	return &mvcc{data: make(map[string][]versionedValue)}
}

func (m *mvcc) Read(key string, timestamp time.Time) (interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	versions, ok := m.data[key]
	if !ok {
		return nil, nil
	}
	for i := len(versions) - 1; i >= 0; i-- {
		if versions[i].timestamp.Before(timestamp) || versions[i].timestamp.Equal(timestamp) {
			return versions[i].value, nil
		}
	}
	return nil, nil
}

func (m *mvcc) Write(key string, value interface{}, timestamp time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.data[key]; !ok {
		m.data[key] = []versionedValue{}
	}
	m.data[key] = append(m.data[key], versionedValue{value, timestamp})
	return nil
}
