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
    fmt.Println("The sum is:", c)
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
	log.Println()

	log.Println("Generated SSA:")
	for _, function := range objFile.SSA.Functions {
		log.Printf("Function: %s", function.Name)
		for _, instruction := range function.Instructions {
			log.Printf("  %s %v -> %s", instruction.Op, instruction.Operands, instruction.Result)
		}
	}
	log.Println()

	// 链接
	linker := &GoLinker{}
	executable := linker.Link([]ObjectFile{objFile})
	log.Println()

	// 输出结果
	log.Println("Compilation and linking completed.")
	log.Printf("Executable size: %d bytes", len(executable.Code))
	log.Println("\nMemory layout:")
	for sym, addr := range executable.MemoryLayout {
		log.Printf("  %s: 0x%x", sym, addr)
	}
}

func init() {
	log.SetFlags(0)
}
