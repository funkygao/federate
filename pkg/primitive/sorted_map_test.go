package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewSortedMap(t *testing.T) {
	data := map[string]interface{}{
		"b": 2,
		"a": 1,
		"c": 3,
	}
	sm := NewSortedMap(data)

	assert.Equal(t, data, sm.data, "The data in SortedMap should be equal to the input map")
}

func TestSortedMap_Keys(t *testing.T) {
	data := map[string]interface{}{
		"b": 2,
		"a": 1,
		"c": 3,
	}
	sm := NewSortedMap(data)
	expectedKeys := []string{"a", "b", "c"}
	keys := sm.Keys()

	assert.Equal(t, expectedKeys, keys, "The keys should be sorted alphabetically")
}

func TestSortedMap_EmptyMap(t *testing.T) {
	data := map[string]interface{}{}
	sm := NewSortedMap(data)
	expectedKeys := []string{}
	keys := sm.Keys()

	assert.Equal(t, expectedKeys, keys, "The keys should be an empty slice for an empty map")
}
