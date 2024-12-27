package main

import (
	"io/ioutil"

	"go/ast"
	"go/parser"
	"go/token"
)

func main() {
	content, _ := ioutil.ReadFile("ast.go")
	src := string(content)

	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, "", src, 0)
	ast.Print(fset, f)
}
