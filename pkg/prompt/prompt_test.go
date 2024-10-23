package prompt

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShouldIgnoreFile(t *testing.T) {
	assert.Equal(t, true, shouldIgnoreFile(".DS_Store"))
	assert.Equal(t, true, shouldIgnoreFile("a/.DS_Store"))
	assert.Equal(t, true, shouldIgnoreFile("./.DS_Store"))
	assert.Equal(t, true, shouldIgnoreFile("a/b/c/.DS_Store"))
	assert.Equal(t, true, shouldIgnoreFile("../.DS_Store"))
	assert.Equal(t, true, shouldIgnoreFile("../a/.DS_Store"))
	assert.Equal(t, false, shouldIgnoreFile("../a/a.java"))
}

func TestIsBinaryFile(t *testing.T) {
	tests := []struct {
		name     string
		content  []byte
		expected bool
	}{
		{
			name: "Text file",
			content: []byte(`#!/bin/bash

# 找到所有扩展名
extensions=$(find . -type f -path '*/src/main/resources/*' | sed -n 's/.*\.\([^.]*\)$/\1/p' | sort | uniq)

# 遍历每个扩展名并找到它们所在的目录
for ext in $extensions; {
    echo "扩展名: $ext"
    find . -type f -path '*/src/main/resources/*' -name "*.$ext" -exec dirname {} \; | sort | uniq
    echo
}`),
			expected: false,
		},
		{
			name:     "Binary file",
			content:  []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F},
			expected: true,
		},
		{
			name:     "Mixed content with more text",
			content:  append([]byte(`This is a text file with some binary data at the end`), 0x00, 0x01, 0x02),
			expected: true,
		},
		{
			name:     "Mixed content with more binary",
			content:  append([]byte{0x00, 0x01, 0x02}, []byte(`This is a text file with some binary data at the end`)...),
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isBinaryFile(tt.content)
			if result != tt.expected {
				t.Errorf("isBinaryFile() = %v, expected %v", result, tt.expected)
			}
		})
	}
}
