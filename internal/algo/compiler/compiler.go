package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
)

type Compiler interface {
	Compile(pkg Package) ObjectFile
}

type GoCompiler struct {
	nextVar int
}

func (c *GoCompiler) Compile(pkg Package) ObjectFile {
	var ssaCode SSACode
	var machineCode []byte
	symbols := make(map[string]int)
	address := 0

	fset := token.NewFileSet()

	for _, file := range pkg.Files {
		log.Printf("Compiling %s...", file.Name)

		// 解析源代码
		astFile, err := parser.ParseFile(fset, file.Name, file.Content, 0)
		if err != nil {
			log.Printf("Error parsing %s: %v", file.Name, err)
			continue
		}

		// 遍历 AST 生成符号表和 SSA
		ast.Inspect(astFile, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.FuncDecl:
				funcName := x.Name.Name
				ssaFunc := c.generateSSAForFunction(x)
				ssaCode.Functions = append(ssaCode.Functions, ssaFunc)

				funcSize := len(ssaFunc.Instructions) * 4 // 假设每条 SSA 指令占 4 字节
				symbols[pkg.Name+"."+funcName] = address
				address += funcSize
				log.Printf("  Found function: %s at address 0x%x, estimated size: %d bytes", funcName, address-funcSize, funcSize)

			case *ast.GenDecl:
				if x.Tok == token.VAR {
					for _, spec := range x.Specs {
						if valueSpec, ok := spec.(*ast.ValueSpec); ok {
							for _, ident := range valueSpec.Names {
								varName := ident.Name
								symbols[pkg.Name+"."+varName] = address
								address += 8 // 假设每个变量占用8字节
								log.Printf("  Found variable: %s at address 0x%x", varName, address-8)
							}
						}
					}
				}
			}

			return true
		})

		// 模拟机器码生成
		machineCode = append(machineCode, []byte("MachineCode for "+file.Name)...)
	}

	return ObjectFile{
		Symbols: symbols,
		Code:    machineCode,
		SSA:     ssaCode,
	}
}

func (c *GoCompiler) generateSSAForFunction(funcDecl *ast.FuncDecl) SSAFunction {
	ssaFunc := SSAFunction{Name: funcDecl.Name.Name}
	c.nextVar = 1 // 重置临时变量计数器

	// 为参数生成 SSA
	for _, field := range funcDecl.Type.Params.List {
		for _, name := range field.Names {
			ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
				Op:       "PARAM",
				Operands: []string{name.Name},
				Result:   c.newTemp(),
			})
		}
	}

	// 为函数体生成 SSA
	c.generateSSAForStmt(&ssaFunc, funcDecl.Body)

	return ssaFunc
}

func (c *GoCompiler) generateSSAForStmt(ssaFunc *SSAFunction, stmt ast.Stmt) {
	switch s := stmt.(type) {
	case *ast.BlockStmt:
		for _, stmt := range s.List {
			c.generateSSAForStmt(ssaFunc, stmt)
		}
	case *ast.AssignStmt:
		c.generateSSAForAssign(ssaFunc, s)
	case *ast.ReturnStmt:
		c.generateSSAForReturn(ssaFunc, s)
	case *ast.ExprStmt:
		c.generateSSAForExpr(ssaFunc, s.X)
	default:
		log.Printf("Unsupported statement type: %T", stmt)
	}
}

func (c *GoCompiler) generateSSAForAssign(ssaFunc *SSAFunction, assign *ast.AssignStmt) {
	if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
		log.Printf("Unsupported assignment: multiple LHS or RHS")
		return
	}

	lhs := assign.Lhs[0].(*ast.Ident).Name
	rhs := c.generateSSAForExpr(ssaFunc, assign.Rhs[0])

	ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
		Op:       "ASSIGN",
		Operands: []string{rhs},
		Result:   lhs,
	})
}

func (c *GoCompiler) generateSSAForReturn(ssaFunc *SSAFunction, ret *ast.ReturnStmt) {
	if len(ret.Results) > 0 {
		result := c.generateSSAForExpr(ssaFunc, ret.Results[0])
		ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
			Op:       "RETURN",
			Operands: []string{result},
		})
	} else {
		ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
			Op: "RETURN",
		})
	}
}

func (c *GoCompiler) generateSSAForExpr(ssaFunc *SSAFunction, expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		left := c.generateSSAForExpr(ssaFunc, e.X)
		right := c.generateSSAForExpr(ssaFunc, e.Y)
		result := c.newTemp()
		ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
			Op:       string(e.Op),
			Operands: []string{left, right},
			Result:   result,
		})
		return result
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return e.Value
	case *ast.CallExpr:
		return c.generateSSAForCall(ssaFunc, e)
	default:
		log.Printf("Unsupported expression type: %T", expr)
		return c.newTemp()
	}
}

func (c *GoCompiler) generateSSAForCall(ssaFunc *SSAFunction, call *ast.CallExpr) string {
	args := make([]string, 0, len(call.Args))
	for _, arg := range call.Args {
		args = append(args, c.generateSSAForExpr(ssaFunc, arg))
	}

	result := c.newTemp()
	funcName := ""
	switch fn := call.Fun.(type) {
	case *ast.Ident:
		funcName = fn.Name
	case *ast.SelectorExpr:
		funcName = fn.Sel.Name
	default:
		log.Printf("Unsupported function call type: %T", call.Fun)
		funcName = "unknown"
	}

	ssaFunc.Instructions = append(ssaFunc.Instructions, SSAInstruction{
		Op:       "CALL",
		Operands: append([]string{funcName}, args...),
		Result:   result,
	})

	return result
}

func (c *GoCompiler) newTemp() string {
	temp := fmt.Sprintf("t%d", c.nextVar)
	c.nextVar++
	return temp
}
