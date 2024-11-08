package ast

import (
	"fmt"
	"time"

	"federate/pkg/ast/parser"
	"federate/pkg/java"
	"github.com/fatih/color"
)

type javaFileConsumer struct {
	parser   *javaParser
	listener parser.Java8ParserListener
}

func (c *javaFileConsumer) Consume(files <-chan fileInfo, errors chan<- error) {
	for file := range files {
		var t0 time.Time
		if c.parser.debug {
			t0 = time.Now()
		}
		err := c.parser.Parse(file.Content, c.listener)
		if c.parser.debug {
			color.Cyan("%80s %v", java.JavaFile2Class(file.Path), time.Since(t0))
			if err != nil {
				color.Red("%s %v", file.Path, err)
			}
		}
		if err != nil {
			errors <- fmt.Errorf("error parsing file %s: %v", file.Path, err)
		}
	}
}
