package insight

import (
	"io/ioutil"
	"log"

	"federate/pkg/ast"
	"federate/pkg/ast/parser"
	"federate/pkg/profiler"
)

func parseDir(dir string, listener parser.Java8ParserListener) error {
	p := prepareParser()
	return p.ParseDirectory(dir, listener)
}

func parseFile(file string, listener parser.Java8ParserListener) error {
	p := prepareParser()
	code, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("%s: %v", file, err)
	}
	return p.ParseJava(string(code), listener)
}

func prepareParser() ast.Parser {
	p := ast.NewParser()
	if debug {
		p.EnableDebug()
	}
	if pprof {
		profiler.Enable()
	}
	return p
}
