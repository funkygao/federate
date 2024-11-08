package listener

import (
	"federate/pkg/ast/parser"
)

func NewMethodCountListner() *MethodCountListener {
	return &MethodCountListener{}
}

type MethodCountListener struct {
	parser.BaseJava8ParserListener

	MethodCount int
}

func (l *MethodCountListener) EnterMethodDeclaration(ctx *parser.MethodDeclarationContext) {
	l.MethodCount++
}
