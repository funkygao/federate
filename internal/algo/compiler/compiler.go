package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"strconv"
)

type Compiler interface {
	Compile(pkg Package) ObjectFile
}

type GoCompiler struct {
	nextVar int
}

func (c *GoCompiler) Compile(pkg Package) ObjectFile {
	var ssaCode SSACode
	var machineCode MachineCode
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

		ast.Print(fset, astFile)

		// 遍历 AST 生成符号表和 SSA
		ast.Inspect(astFile, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.FuncDecl:
				funcName := x.Name.Name
				ssaFunc := c.generateSSAForFunction(x)
				ssaCode.Functions = append(ssaCode.Functions, ssaFunc)

				funcMachineCode := c.generateMachineCode(ssaFunc)
				machineCode.Instructions = append(machineCode.Instructions, funcMachineCode.Instructions...)

				funcSize := len(funcMachineCode.Instructions) * 4 // 假设每条机器指令占 4 字节
				symbols[pkg.Name+"."+funcName] = address
				address += funcSize
				log.Printf("  Found function: %s at address 0x%x, size: %d bytes", funcName, address-funcSize, funcSize)

			case *ast.GenDecl:
				if x.Tok == token.VAR {
					for _, spec := range x.Specs {
						if valueSpec, ok := spec.(*ast.ValueSpec); ok {
							for _, ident := range valueSpec.Names {
								varName := ident.Name
								symbols[pkg.Name+"."+varName] = address
								address += 8 // 假设每个变量占用8字节
								log.Printf("  Found variable: %s at address 0x%x", varName, address-8)

								// 为全局变量添加一条机器指令（仅作为占位符）
								machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
									Opcode:   "DATA",
									Operands: []string{varName, "8"}, // 8 表示 8 字节
								})
							}
						}
					}
				}
			}
			return true
		})
	}

	return ObjectFile{
		Symbols: symbols,
		Code:    machineCode,
		SSA:     ssaCode,
	}
}

func (c *GoCompiler) generateMachineCode(ssaFunc SSAFunction) MachineCode {
	var machineCode MachineCode
	registers := []string{"rax", "rbx", "rcx", "rdx"}
	regMap := make(map[string]string)

	for _, instr := range ssaFunc.Instructions {
		switch instr.Op {
		case "PARAM":
			reg := c.allocateRegister(registers, regMap)
			regMap[instr.Result] = reg
			machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
				Opcode:   "MOV",
				Operands: []string{reg, instr.Operands[0]},
			})
		case "ASSIGN":
			reg := c.allocateRegister(registers, regMap)
			regMap[instr.Result] = reg
			machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
				Opcode:   "MOV",
				Operands: []string{reg, c.getOperand(instr.Operands[0], regMap)},
			})
		case "+", "-", "*", "/":
			reg1 := c.getOperand(instr.Operands[0], regMap)
			reg2 := c.getOperand(instr.Operands[1], regMap)
			resultReg := c.allocateRegister(registers, regMap)
			regMap[instr.Result] = resultReg
			machineCode.Instructions = append(machineCode.Instructions,
				MachineInstruction{Opcode: "MOV", Operands: []string{resultReg, reg1}},
			)
			opcode := ""
			switch instr.Op {
			case "+":
				opcode = "ADD"
			case "-":
				opcode = "SUB"
			case "*":
				opcode = "IMUL"
			case "/":
				opcode = "IDIV"
			}
			machineCode.Instructions = append(machineCode.Instructions,
				MachineInstruction{Opcode: opcode, Operands: []string{resultReg, reg2}},
			)
		case "CALL":
			for i, arg := range instr.Operands[1:] {
				machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
					Opcode:   "MOV",
					Operands: []string{fmt.Sprintf("arg%d", i), c.getOperand(arg, regMap)},
				})
			}
			machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
				Opcode:   "CALL",
				Operands: []string{instr.Operands[0]},
			})
			if instr.Result != "" {
				resultReg := c.allocateRegister(registers, regMap)
				regMap[instr.Result] = resultReg
				machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
					Opcode:   "MOV",
					Operands: []string{resultReg, "rax"},
				})
			}
		case "RETURN":
			if len(instr.Operands) > 0 {
				machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
					Opcode:   "MOV",
					Operands: []string{"rax", c.getOperand(instr.Operands[0], regMap)},
				})
			}
			machineCode.Instructions = append(machineCode.Instructions, MachineInstruction{
				Opcode: "RET",
			})
		default:
			log.Printf("Unsupported SSA operation: %s", instr.Op)
		}
	}

	return machineCode
}

func (c *GoCompiler) allocateRegister(registers []string, regMap map[string]string) string {
	for _, reg := range registers {
		used := false
		for _, allocatedReg := range regMap {
			if allocatedReg == reg {
				used = true
				break
			}
		}
		if !used {
			return reg
		}
	}
	return registers[0] // 如果所有寄存器都被使用，简单地返回第一个（在实际编译器中，这里会进行寄存器溢出处理）
}

func (c *GoCompiler) getOperand(op string, regMap map[string]string) string {
	if reg, ok := regMap[op]; ok {
		return reg
	}
	if _, err := strconv.Atoi(op); err == nil {
		return op // 如果是数字，直接返回
	}
	return op // 如果不是寄存器也不是数字，假设是内存地址或标签
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
