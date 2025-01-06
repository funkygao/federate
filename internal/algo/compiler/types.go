package main

type SourceFile struct {
	Name    string
	Content string
}

type Package struct {
	Name  string
	Files []SourceFile
}

type ObjectFile struct {
	Symbols map[string]int
	Code    MachineCode
	SSA     SSACode
}

type Executable struct {
	Code         MachineCode
	MemoryLayout map[string]uintptr
}

type SSAInstruction struct {
	Op       string
	Operands []string
	Result   string
}

type SSAFunction struct {
	Name         string
	Instructions []SSAInstruction
}

// 具有SSA(Static Single Assignment)特性的中间代码
// SSA is a type of intermediate representation (IR) where each variable is assigned exactly once.
// SSA 的主要作用是对代码进行优化
type SSACode struct {
	Functions []SSAFunction
}

type MachineInstruction struct {
	Opcode   string
	Operands []string
}

type MachineCode struct {
	Instructions []MachineInstruction
}
