package ast

import (
	"fmt"
	"runtime"
	"sync"

	"federate/pkg/ast/parser"
	"github.com/antlr4-go/antlr/v4"
)

// Parser 定义了 Java8 解析器的接口
type Parser interface {
	Parse(javaSrc string, listener parser.Java8ParserListener) error

	ParseDirectory(dir string, listener parser.Java8ParserListener) error
}

func NewParser() Parser {
	return &javaParser{}
}

type javaParser struct {
	lexer  *parser.Java8Lexer
	parser *parser.Java8Parser
}

func (p *javaParser) Parse(javaSrc string, listener parser.Java8ParserListener) error {
	errorListener := newErrorListener()

	// 设置词法分析器及其错误监听
	p.lexer = parser.NewJava8Lexer(antlr.NewInputStream(javaSrc))
	p.lexer.RemoveErrorListeners()
	p.lexer.AddErrorListener(errorListener)

	// 设置语法分析器及其错误监听
	p.parser = parser.NewJava8Parser(antlr.NewCommonTokenStream(p.lexer, antlr.TokenDefaultChannel))
	p.parser.RemoveErrorListeners()
	p.parser.AddErrorListener(errorListener)

	// 获取编译单元（根节点）
	tree := p.parser.CompilationUnit()
	if errorListener.HasError() {
		return fmt.Errorf("parsing errors: %v", errorListener.Error())
	}

	// 遍历语法树
	antlr.ParseTreeWalkerDefault.Walk(listener, tree)

	return nil
}

func (p *javaParser) ParseDirectory(dir string, listener parser.Java8ParserListener) error {
	numWorkers := runtime.NumCPU()

	producer := &JavaFileProducer{}
	filesChan, producerErrors := producer.Produce(dir)

	consumer := &JavaFileConsumer{parser: p, listener: listener}

	errorsChan := make(chan error)
	var wg sync.WaitGroup

	// 启动 consumer goroutines
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			consumer.Consume(filesChan, errorsChan)
		}()
	}

	// 收集 producer 错误
	go func() {
		for err := range producerErrors {
			errorsChan <- err
		}
	}()

	// 等待所有 consumer 完成并关闭 errors channel
	go func() {
		wg.Wait()
		close(errorsChan)
	}()

	// 收集所有错误
	var errors []error
	for err := range errorsChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("encountered %d errors during parsing: %v", len(errors), errors)
	}

	return nil
}
