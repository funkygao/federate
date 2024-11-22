package code

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestJavaFileLines(t *testing.T) {
	content := "line1\nline2\nline3"
	j := &JavaFile{content: content}

	expected := []string{"line1", "line2", "line3"}

	// 第一次调用
	lines1 := j.RawLines()
	assert.Equal(t, expected, lines1, "First call should return correct lines")

	// 检查 codeLines 是否被设置
	assert.NotNil(t, j.cachedLines, "cacheLines should be set after first call")

	// 第二次调用
	lines2 := j.RawLines()
	assert.Equal(t, expected, lines2, "Second call should return the same lines")

	// 验证两次调用返回相同的切片（地址相同）
	assert.Same(t, &lines1[0], &lines2[0], "Subsequent calls should return the same slice")

	// 内容修改，缓存自动更新
	j.UpdateContent("newline1\nnewline2")
	lines3 := j.RawLines()
	assert.NotEqual(t, expected, lines3, "After content change, return updated lines")
	assert.Equal(t, []string{"newline1", "newline2"}, j.RawLines())
}
