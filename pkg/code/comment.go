package code

import (
	"strings"
)

// 跟踪 java 文件的注释行状态，主要是跨多行
type CommentTracker struct {
	inMultiLineComment bool
}

func NewCommentTracker() *CommentTracker {
	return &CommentTracker{inMultiLineComment: false}
}

func (c *CommentTracker) InComment(line string) bool {
	trimmedLine := strings.TrimSpace(line)
	if strings.HasPrefix(trimmedLine, "/*") {
		c.inMultiLineComment = true
	}
	if strings.HasSuffix(trimmedLine, "*/") {
		c.inMultiLineComment = false
		return true
	}
	return c.inMultiLineComment || strings.HasPrefix(trimmedLine, "//")
}
