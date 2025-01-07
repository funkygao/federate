package mybatis

import (
	"fmt"
	"log"
	"strings"

	"github.com/xwb1989/sqlparser"
)

type TableIndexRecommendation struct {
	Table             string
	FieldCombinations map[string]int

	JoinFields    map[string]int
	WhereFields   map[string]int
	GroupByFields map[string]int
	OrderByFields map[string]int
}

type UnparsableSQL struct {
	Stmt  Statement
	Error error
}

type SQLAnalyzer struct {
	IgnoredFields map[string]bool
	DB            *DB

	StatementsByType map[string][]*Statement

	SQLTypes             map[string]int
	Tables               map[string]int
	Fields               map[string]int
	UnionOperations      int
	SubQueries           int
	AggregationFuncs     map[string]map[string]int // count, min, max, etc, key1 is OpType
	DistinctQueries      int
	OrderByOperations    int
	LimitOperations      int
	LimitWithOffset      int
	LimitWithoutOffset   int
	JoinOperations       int
	JoinTypes            map[string]int
	JoinTableCounts      map[int]int
	JoinConditions       map[string]int
	IndexHints           map[string]int
	ParsedOK             int
	TimeoutStatements    map[string]int
	IndexRecommendations map[string]*TableIndexRecommendation

	// metrics
	TableUsage             map[string]*TableUsage
	TableRelations         []TableRelation
	PerformanceBottlenecks []string
	ComplexQueries         []SQLComplexity
	OptimisticLocks        []*Statement
	ReuseOpportunities     []string

	UnknownFragments map[string][]SqlFragmentRef
	UnparsableSQL    []UnparsableSQL
}

func NewSQLAnalyzer(ignoredFields []string, DB *DB) *SQLAnalyzer {
	sa := &SQLAnalyzer{
		DB:                DB,
		IgnoredFields:     make(map[string]bool),
		SQLTypes:          make(map[string]int),
		Tables:            make(map[string]int),
		Fields:            make(map[string]int),
		AggregationFuncs:  make(map[string]map[string]int),
		JoinTypes:         make(map[string]int),
		JoinTableCounts:   make(map[int]int),
		JoinConditions:    make(map[string]int),
		IndexHints:        make(map[string]int),
		TimeoutStatements: make(map[string]int),
		StatementsByType:  make(map[string][]*Statement),

		IndexRecommendations: make(map[string]*TableIndexRecommendation),

		UnparsableSQL:    []UnparsableSQL{},
		UnknownFragments: make(map[string][]SqlFragmentRef),
	}
	sa.JoinConditions["ON"] = 0
	sa.JoinConditions["USING"] = 0
	for _, field := range ignoredFields {
		sa.IgnoredFields[field] = true
	}
	return sa
}

func (sa *SQLAnalyzer) AnalyzeStmt(s Statement) error {
	if s.Timeout > 0 {
		timeoutKey := fmt.Sprintf("%ds", s.Timeout)
		sa.TimeoutStatements[timeoutKey]++
	}

	var parseErrors []error
	for _, subStmt := range s.SplitSQL() {
		stmt, err := sqlparser.Parse(subStmt)
		if err != nil {
			parseErrors = append(parseErrors, fmt.Errorf("error parsing SQL: %v. SQL: %s", err, subStmt))
			continue
		}

		s.SubSQL = append(s.SubSQL, subStmt)

		sa.ParsedOK++
		sa.StatementsByType[s.Type] = append(sa.StatementsByType[s.Type], &s)

		stmtID, err := sa.DB.InsertStatement(&s)
		if err != nil {
			log.Printf("Error inserting statement: %v", err)
		}

		switch stmt := stmt.(type) {
		case *sqlparser.Select:
			if !isSelectFromDual(stmt) {
				s.SetPrimarySQL(subStmt)
				sa.analyzeSelect(stmt, s, stmtID)
			}

		case *sqlparser.Insert:
			s.SetPrimarySQL(subStmt)
			if stmt.Action == sqlparser.ReplaceStr {
				sa.analyzeReplace(stmt, s, stmtID)
			} else {
				sa.analyzeInsert(stmt, s, stmtID)
			}

		case *sqlparser.Update:
			s.SetPrimarySQL(subStmt)
			sa.analyzeUpdate(stmt, s, stmtID)

		case *sqlparser.Delete:
			s.SetPrimarySQL(subStmt)
			sa.analyzeDelete(stmt, s, stmtID)

		case *sqlparser.Union:
			s.SetPrimarySQL(subStmt)
			sa.analyzeUnion(stmt, s, stmtID)

		case *sqlparser.Set:
			sa.analyzeSet(stmt, stmtID)

		default:
			log.Printf("Unhandled SQL type: %T\nSQL: %s", stmt, subStmt)
		}
	}

	if len(parseErrors) > 0 {
		return fmt.Errorf("encountered %d parse errors: %v", len(parseErrors), parseErrors)
	}

	return nil
}

func (sa *SQLAnalyzer) IncrementAggregationFunc(opType, funcName string) {
	if _, present := sa.AggregationFuncs[opType]; !present {
		sa.AggregationFuncs[opType] = make(map[string]int)
	}
	sa.AggregationFuncs[opType][strings.ToUpper(funcName)]++
}
