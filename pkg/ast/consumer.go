package ast

import (
	"fmt"

	"federate/pkg/ast/parser"
)

type javaFileConsumer struct {
	parser   Parser
	listener parser.Java8ParserListener
}

func (c *javaFileConsumer) Consume(files <-chan fileInfo, errors chan<- error) {
	for file := range files {
		if err := c.parser.Parse(file.Content, c.listener); err != nil {
			errors <- fmt.Errorf("error parsing file %s: %v", file.Path, err)
		}
	}
}
