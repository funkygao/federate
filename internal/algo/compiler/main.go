package main

import (
	"log"
)

func main() {
	pkg := Package{
		Name: "example",
		Files: []SourceFile{
			{
				Name: "file1.go",
				Content: `package example

import "fmt"

var GlobalVar int

func Add(a, b int) int {
    c := a + b
    return c
}

func PrintHello() {
    fmt.Println("Hello, World!")
}`,
			},
			{
				Name: "file2.go",
				Content: `package example

var (
    AnotherVar float64
    YetAnotherVar string
)

func Sub(a, b int) int {
    return a - b
}`,
			},
		},
	}

	// 编译
	compiler := &GoCompiler{}
	objFile := compiler.Compile(pkg)

	// 输出 SSA
	log.Println("\nGenerated SSA:")
	for _, function := range objFile.SSA.Functions {
		log.Printf("Function: %s", function.Name)
		for _, instruction := range function.Instructions {
			log.Printf("  %s %v -> %s", instruction.Op, instruction.Operands, instruction.Result)
		}
	}

	// 输出机器码
	log.Println("\nGenerated Machine Code:")
	for i, instruction := range objFile.Code.Instructions {
		log.Printf("  0x%04x: %s %v", i*4, instruction.Opcode, instruction.Operands)
	}

	// 链接
	linker := &GoLinker{}
	executable := linker.Link([]ObjectFile{objFile})
	log.Println()

	// 输出最终的可执行代码
	log.Println("\nFinal Executable Code:")
	for i, instruction := range executable.Code.Instructions {
		log.Printf("  0x%04x: %s %v", i*4, instruction.Opcode, instruction.Operands)
	}

	log.Println("\nMemory layout:")
	for sym, addr := range executable.MemoryLayout {
		log.Printf("  %s: 0x%x", sym, addr)
	}
}

func init() {
	log.SetFlags(0)
}
