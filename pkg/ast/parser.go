package ast

import (
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"sync"

	"federate/pkg/ast/parser"
	"github.com/antlr4-go/antlr/v4"
)

// Parser 定义了 Java8 解析器的接口
type Parser interface {
	// Parse a single java source content.
	Parse(javaSrc string, listener parser.Java8ParserListener) error

	// Recursively parse java sources from a directory.
	ParseDirectory(dir string, listener parser.Java8ParserListener) error

	EnableDebug()
	EnablePprof(port string)
}

func NewParser() Parser {
	return &javaParser{}
}

type javaParser struct {
	debug        bool
	pprofEnabled bool

	lexer  *parser.Java8Lexer
	parser *parser.Java8Parser
}

func (p *javaParser) EnableDebug() {
	p.debug = true
}

func (p *javaParser) EnablePprof(port string) {
	p.pprofEnabled = true
	go func() {
		log.Println(http.ListenAndServe("localhost:"+port, nil))
	}()
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

	producer := &javaFileProducer{parser: p}
	filesChan := producer.Produce(dir)

	consumer := &javaFileConsumer{parser: p, listener: listener}

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

	// 等待所有 consumer 完成并关闭 errors channel
	go func() {
		wg.Wait()
		close(errorsChan)
	}()

	var errors []error
	for err := range errorsChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("encountered %d errors during parsing: %v", len(errors), errors)
	}

	return nil
}
