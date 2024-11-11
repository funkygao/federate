package insight

import (
	"federate/pkg/ast"
	"federate/pkg/ast/parser"
	"federate/pkg/profiler"
)

func parseDir(dir string, listener parser.Java8ParserListener) error {
	p := ast.NewParser()
	if debug {
		p.EnableDebug()
	}
	if pprof {
		profiler.Enable()
	}

	return p.ParseDirectory(dir, listener)
}
