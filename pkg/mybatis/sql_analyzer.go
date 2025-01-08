package mybatis

import (
	"fmt"
	"log"

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

	StatementsByTag map[string][]*Statement

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
		StatementsByTag:   make(map[string][]*Statement),

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

	sa.addStatement(&s)

	if err := s.Analyze(); err != nil {
		sa.UnparsableSQL = append(sa.UnparsableSQL, UnparsableSQL{s, err})
		return err
	}

	sa.updateAggregatedMetrics(&s)

	for _, subSQL := range s.SplitSQL() {
		stmt, err := sqlparser.Parse(subSQL)
		if err != nil {
			continue
		}

		sa.ParsedOK++

		stmtID, err := sa.DB.InsertStatement(&s)
		if err != nil {
			log.Printf("Error inserting statement: %v", err)
		}

		switch stmt := stmt.(type) {
		case *sqlparser.Select:
			if !isSelectFromDual(stmt) {
				sa.analyzeSelect(stmt, s, stmtID)
			}

		case *sqlparser.Insert:
			if stmt.Action == sqlparser.ReplaceStr {
				sa.analyzeReplace(stmt, s, stmtID)
			} else {
				sa.analyzeInsert(stmt, s, stmtID)
			}

		case *sqlparser.Update:
			sa.analyzeUpdate(stmt, s, stmtID)

		case *sqlparser.Delete:
			sa.analyzeDelete(stmt, s, stmtID)

		case *sqlparser.Union:
			sa.analyzeUnion(stmt, s, stmtID)

		case *sqlparser.Set:
			sa.analyzeSet(stmt, stmtID)

		default:
			log.Printf("Unhandled SQL type: %T\nSQL: %s", stmt, subSQL)
		}
	}

	return nil
}

func (sa *SQLAnalyzer) updateAggregatedMetrics(stmt *Statement) {
	sa.ComplexQueries = append(sa.ComplexQueries, stmt.Complexity)

	s := stmt.Metadata

	sa.SQLTypes[s.SQLType]++
	for _, table := range s.Tables {
		sa.Tables[table]++
	}
	for _, field := range s.Fields {
		sa.Fields[field]++
	}
	sa.UnionOperations += s.UnionOperations
	sa.SubQueries += s.SubQueries
	for funcName, count := range s.AggregationFuncs {
		if sa.AggregationFuncs[s.SQLType] == nil {
			sa.AggregationFuncs[s.SQLType] = make(map[string]int)
		}
		sa.AggregationFuncs[s.SQLType][funcName] += count
	}
	if s.HasDistinct {
		sa.DistinctQueries++
	}
	if s.HasOrderBy {
		sa.OrderByOperations++
	}
	if s.HasLimit {
		sa.LimitOperations++
		if s.HasOffset {
			sa.LimitWithOffset++
		} else {
			sa.LimitWithoutOffset++
		}
	}
	sa.JoinOperations += s.JoinOperations
	for joinType, count := range s.JoinTypes {
		sa.JoinTypes[joinType] += count
	}
	sa.JoinTableCounts[s.JoinTableCount]++
	for condition, count := range s.JoinConditions {
		sa.JoinConditions[condition] += count
	}
	for hint, count := range s.IndexHints {
		sa.IndexHints[hint] += count
	}

	if stmt.HasOptimisticLocking() {
		sa.OptimisticLocks = append(sa.OptimisticLocks, stmt)
	}
}

// WalkStatements 遍历 SQLAnalyzer 中的所有语句
func (sa *SQLAnalyzer) WalkStatements(walkFn func(tag string, stmt *Statement) error) error {
	for tag, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			if err := walkFn(tag, stmt); err != nil {
				return err
			}
		}
	}
	return nil
}

func (sa *SQLAnalyzer) addStatement(s *Statement) {
	// 确保 StatementsByTag 已初始化
	if sa.StatementsByTag == nil {
		sa.StatementsByTag = make(map[string][]*Statement)
	}

	sa.StatementsByTag[s.Tag] = append(sa.StatementsByTag[s.Tag], s)
}
