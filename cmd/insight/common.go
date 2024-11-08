package insight

import (
	"federate/pkg/ast"
	"federate/pkg/ast/parser"
)

func parseDir(dir string, listener parser.Java8ParserListener) error {
	p := ast.NewParser()
	if debug {
		p.Debug()
	}

	return p.ParseDirectory(dir, listener)
}
