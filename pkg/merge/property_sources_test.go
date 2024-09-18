package merge

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnflattenYamlMap(t *testing.T) {
	cm := &PropertySourcesManager{}

	tests := []struct {
		name     string
		input    map[string]interface{}
		expected map[string]interface{}
	}{
		{
			name: "simple nested map",
			input: map[string]interface{}{
				"a.b.c": "d",
			},
			expected: map[string]interface{}{
				"a": map[string]interface{}{
					"b": map[string]interface{}{
						"c": "d",
					},
				},
			},
		},
		{
			name: "flat map",
			input: map[string]interface{}{
				"a": "b",
				"c": "d",
			},
			expected: map[string]interface{}{
				"a": "b",
				"c": "d",
			},
		},
		{
			name: "mixed nested map",
			input: map[string]interface{}{
				"a.b": "c",
				"d":   "e",
			},
			expected: map[string]interface{}{
				"a": map[string]interface{}{
					"b": "c",
				},
				"d": "e",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cm.unflattenYamlMap(tt.input)
			assert.Equal(t, tt.expected, result, "they should be equal")
		})
	}
}
