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
	Code    []byte
	SSA     SSACode
}

type Executable struct {
	Code         []byte
	MemoryLayout map[string]uintptr
}
