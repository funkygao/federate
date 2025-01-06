package main

type SSAInstruction struct {
	Op       string
	Operands []string
	Result   string
}

type SSAFunction struct {
	Name         string
	Instructions []SSAInstruction
}

// SSA中间代码
// Static Single Assignment
type SSACode struct {
	Functions []SSAFunction
}
