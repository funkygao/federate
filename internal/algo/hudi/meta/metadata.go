package meta

import "sync"

type Metadata struct {
	mu   sync.RWMutex
	data map[string]string
}

func NewMetadata() *Metadata {
	return &Metadata{
		data: make(map[string]string),
	}
}

func (m *Metadata) Set(key, value string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = value
}

func (m *Metadata) Get(key string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	value, ok := m.data[key]
	return value, ok
}

func (m *Metadata) GetAll() map[string]string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make(map[string]string)
	for k, v := range m.data {
		result[k] = v
	}
	return result
}
