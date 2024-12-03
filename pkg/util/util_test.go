package util

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestContains(t *testing.T) {
	assert.True(t, Contains("a", []string{"a"}))
	assert.True(t, Contains("a", []string{"a", "b"}))
	assert.True(t, Contains("a", []string{"c", "a", "b"}))
	assert.False(t, Contains("m", []string{"c", "a", "b"}))
}

func TestFileExists(t *testing.T) {
	assert := assert.New(t)

	// 使用 t.TempDir() 创建临时目录
	tempDir := t.TempDir()

	// 测试文件存在的情况
	existingFile := filepath.Join(tempDir, "existing.txt")
	err := os.WriteFile(existingFile, []byte("hello"), 0644)
	assert.NoError(err, "创建测试文件时出错")

	assert.True(FileExists(existingFile), "应该检测到存在的文件")

	// 测试文件不存在的情况
	nonExistingFile := filepath.Join(tempDir, "non_existing.txt")
	assert.False(FileExists(nonExistingFile), "不应该检测到不存在的文件")

	// 测试目录的情况
	assert.False(FileExists(tempDir), "不应该将目录视为存在的文件")
}

func TestMapSortedStringKeys(t *testing.T) {
	m := map[string]interface{}{
		"b": 3,
		"a": 1,
		"c": 2,
	}
	assert.Equal(t, []string{"a", "b", "c"}, MapSortedStringKeys(m))
}

func TestBeautify(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected string
	}{
		{
			name:     "Empty map",
			input:    map[string]interface{}{},
			expected: "{}",
		},
		{
			name: "Simple map",
			input: map[string]interface{}{
				"name": "John",
				"age":  30,
			},
			expected: `{
  "age": 30,
  "name": "John"
}`,
		},
		{
			name: "Nested map",
			input: map[string]interface{}{
				"person": map[string]interface{}{
					"name": "Alice",
					"age":  25,
				},
				"city": "New York",
			},
			expected: `{
  "city": "New York",
  "person": {
    "age": 25,
    "name": "Alice"
  }
}`,
		},
		{
			name:  "Array",
			input: []string{"apple", "banana", "cherry"},
			expected: `[
  "apple",
  "banana",
  "cherry"
]`,
		},
	}

	for _, tt := range tests {
		assert.Equal(t, tt.expected, Beautify(tt.input))
	}
}
