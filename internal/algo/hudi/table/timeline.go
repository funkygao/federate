package table

import (
	"sort"
	"sync"
	"time"
)

type Timeline struct {
	mu       sync.RWMutex
	instants []Instant
}

func NewTimeline() *Timeline {
	return &Timeline{
		instants: make([]Instant, 0),
	}
}

func (t *Timeline) AddInstant(instant Instant) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.instants = append(t.instants, instant)
	sort.Slice(t.instants, func(i, j int) bool {
		return t.instants[i].Timestamp.Before(t.instants[j].Timestamp)
	})
}

func (t *Timeline) GetInstantsAfter(timestamp time.Time) []Instant {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var result []Instant
	for _, instant := range t.instants {
		if instant.Timestamp.After(timestamp) {
			result = append(result, instant)
		}
	}
	return result
}
