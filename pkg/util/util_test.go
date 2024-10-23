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
