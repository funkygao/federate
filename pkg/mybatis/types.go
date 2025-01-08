package mybatis

import (
	"federate/pkg/primitive"
)

type TableIndexRecommendation struct {
	Table             string
	FieldCombinations map[string]int
	JoinFields        map[string]int
	WhereFields       map[string]int
	GroupByFields     map[string]int
	OrderByFields     map[string]int
}

type UnparsableSQL struct {
	Stmt  Statement
	Error error
}

type TableUsage struct {
	Name              string
	UseCount          int // total usage count
	InSelect          int
	InDelete          int
	BatchInsert       int
	SingleInsert      int
	BatchUpdate       int
	SingleUpdate      int
	InsertOnDuplicate int
}

type TableRelation struct {
	Table1        string
	Table2        string
	JoinType      string
	JoinCondition string
}

type SQLComplexity struct {
	Filename    string
	StatementID string
	Score       int
	Reasons     *primitive.StringSet
}

type SqlFragmentRef struct {
	Refid  string
	StmtID string
}
