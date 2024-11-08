package ast

import (
	"fmt"

	"federate/pkg/ast/parser"
)

// Consumer 定义了消费者接口
type Consumer interface {
	Consume(files <-chan FileInfo, errors chan<- error)
}

type JavaFileConsumer struct {
	parser   Parser
	listener parser.Java8ParserListener
}

func (c *JavaFileConsumer) Consume(files <-chan FileInfo, errors chan<- error) {
	for file := range files {
		if err := c.parser.Parse(file.Content, c.listener); err != nil {
			errors <- fmt.Errorf("error parsing file %s: %v", file.Path, err)
		}
	}
}
