package primitive

import (
	"sort"
)

// SortedMap is a structure that holds a map and provides methods to sort and log it
type SortedMap struct {
	data map[string]interface{}
}

// NewSortedMap creates a new SortedMap from a given map
func NewSortedMap(m map[string]interface{}) *SortedMap {
	return &SortedMap{data: m}
}

// Keys returns the sorted keys of the map
func (sm *SortedMap) Keys() []string {
	keys := make([]string, 0, len(sm.data))
	for key := range sm.data {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
