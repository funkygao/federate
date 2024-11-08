package ast

import (
	"fmt"
	"strings"

	"github.com/antlr4-go/antlr/v4"
)

// errorListener 是一个自定义的错误监听器
type errorListener struct {
	*antlr.DefaultErrorListener
	errors []string
}

func newErrorListener() *errorListener {
	return &errorListener{DefaultErrorListener: antlr.NewDefaultErrorListener(), errors: make([]string, 0)}
}

func (l *errorListener) SyntaxError(recognizer antlr.Recognizer, offendingSymbol interface{}, line, column int, msg string, e antlr.RecognitionException) {
	l.errors = append(l.errors, fmt.Sprintf("line %d:%d %s", line, column, msg))
}

func (l *errorListener) HasError() bool {
	return len(l.errors) > 0
}

func (l *errorListener) Error() string {
	return strings.Join(l.errors, "; ")
}
