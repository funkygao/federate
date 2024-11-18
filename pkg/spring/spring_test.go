package spring

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUpdateMap_RuleByFileName(t *testing.T) {
	tests := []struct {
		name      string
		updateMap UpdateMap
		fileName  string
		expected  map[string]string
	}{
		{
			name: "Match component",
			updateMap: UpdateMap{
				"component1": {"key1": "value1", "key2": "value2"},
				"component2": {"key3": "value3"},
			},
			fileName: "/path/to/component1/file.xml",
			expected: map[string]string{"key1": "value1", "key2": "value2"},
		},
		{
			name: "No match",
			updateMap: UpdateMap{
				"component1": {"key1": "value1"},
				"component2": {"key2": "value2"},
			},
			fileName: "/path/to/component3/file.xml",
			expected: nil,
		},
		{
			name:      "Empty UpdateMap",
			updateMap: UpdateMap{},
			fileName:  "/path/to/component1/file.xml",
			expected:  nil,
		},
		{
			name: "Multiple matches, return first",
			updateMap: UpdateMap{
				"component1": {"key1": "value1"},
				"comp":       {"key2": "value2"},
			},
			fileName: "/path/to/component1/file.xml",
			expected: map[string]string{"key1": "value1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.updateMap.RuleByFileName(tt.fileName)
			assert.Equal(t, tt.expected, result, "RuleByFileName() result not as expected")
		})
	}
}
