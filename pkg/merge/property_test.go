package merge

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUpdateRequestMappingInFile(t *testing.T) {
	cm := NewPropertyManager(nil)

	testCases := []struct {
		name        string
		input       string
		contextPath string
		expected    string
	}{
		{
			name:        "Simple RequestMapping",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/api/users")`,
		},
		{
			name:        "RequestMapping with value",
			input:       `@RequestMapping(value = "/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping(value = "/wms-stock/api/users")`,
		},
		{
			name:        "Multiple RequestMappings",
			input:       `@RequestMapping("/api/users")\n@RequestMapping("/api/posts")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/api/users")\n@RequestMapping("/wms-stock/api/posts")`,
		},
		{
			name:        "RequestMapping with existing context path",
			input:       `@RequestMapping("/wms-stock/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/wms-stock/api/users")`,
		},
		{
			name:        "No RequestMapping",
			input:       `public class UserController {}`,
			contextPath: "/wms-stock",
			expected:    `public class UserController {}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := cm.updateRequestMappingInFile(tc.input, tc.contextPath)
			if result != tc.expected {
				t.Errorf("Expected:\n%s\nGot:\n%s", tc.expected, result)
			}
		})
	}
}

func TestUpdateRequestMappingInFile_EdgeCases(t *testing.T) {
	cm := NewPropertyManager(nil)

	testCases := []struct {
		name        string
		input       string
		contextPath string
		expected    string
	}{
		{
			name:        "Empty input",
			input:       "",
			contextPath: "/myapp",
			expected:    "",
		},
		{
			name:        "Empty context path",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "",
			expected:    `@RequestMapping("/api/users")`,
		},
		{
			name:        "Context path without leading slash",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "myapp",
			expected:    `@RequestMapping("myapp/api/users")`,
		},
		{
			name:        "RequestMapping with regex path",
			input:       `@RequestMapping("/api/{userId:[0-9]+}")`,
			contextPath: "/myapp",
			expected:    `@RequestMapping("/myapp/api/{userId:[0-9]+}")`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := cm.updateRequestMappingInFile(tc.input, tc.contextPath)
			assert.Equal(t, tc.expected, result, "The updated content should match the expected output")
		})
	}
}

func TestUnflattenYamlMap(t *testing.T) {
	cm := &PropertyManager{}

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
