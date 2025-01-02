package mybatis

import (
	"fmt"
	"log"
	"regexp"
	"sort"
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

	SQLTypes           map[string]int
	Tables             map[string]int
	Fields             map[string]int
	ComplexQueries     int
	JoinOperations     int
	UnionOperations    int
	SubQueries         int
	AggregationFuncs   map[string]int // count, min, max, etc
	DistinctQueries    int
	OrderByOperations  int
	LimitOperations    int
	JoinTypes          map[string]int
	ParsedOK           int
	BatchInserts       int
	BatchInsertColumns map[string]int

	IndexRecommendations map[string]*TableIndexRecommendation

	UnknownFragments map[string][]SqlFragmentRef
	UnparsableSQL    []UnparsableSQL
}

func NewSQLAnalyzer(ignoredFields []string) *SQLAnalyzer {
	sa := &SQLAnalyzer{
		IgnoredFields:      make(map[string]bool),
		SQLTypes:           make(map[string]int),
		Tables:             make(map[string]int),
		Fields:             make(map[string]int),
		AggregationFuncs:   make(map[string]int),
		JoinTypes:          make(map[string]int),
		BatchInsertColumns: make(map[string]int),

		IndexRecommendations: make(map[string]*TableIndexRecommendation),

		UnparsableSQL:    []UnparsableSQL{},
		UnknownFragments: make(map[string][]SqlFragmentRef),
	}
	for _, field := range ignoredFields {
		sa.IgnoredFields[field] = true
	}
	return sa
}

func (sa *SQLAnalyzer) Visit(xmlPath string, unknowns []SqlFragmentRef) {
	sa.UnknownFragments[xmlPath] = unknowns
}

func (sa *SQLAnalyzer) AnalyzeStmt(s Statement) error {
	stmt, err := sqlparser.Parse(s.ParseableSQL)
	if err != nil {
		sa.UnparsableSQL = append(sa.UnparsableSQL, UnparsableSQL{
			Stmt:  s,
			Error: err,
		})
		return err
	}

	sa.ParsedOK++

	switch stmt := stmt.(type) {
	case *sqlparser.Select:
		sa.analyzeSelect(stmt)
	case *sqlparser.Insert:
		if stmt.Action == sqlparser.ReplaceStr {
			sa.analyzeReplace(stmt)
		} else {
			sa.analyzeInsert(stmt)
		}
	case *sqlparser.Update:
		sa.analyzeUpdate(stmt)
	case *sqlparser.Delete:
		sa.analyzeDelete(stmt)
	case *sqlparser.Union:
		sa.analyzeUnion(stmt)
	default:
		log.Printf("Unhandled SQL type: %T\nSQL: %s", stmt, s.ParseableSQL)
	}

	// Unify aggregation function names to uppercase
	for k, v := range sa.AggregationFuncs {
		delete(sa.AggregationFuncs, k)
		sa.AggregationFuncs[strings.ToUpper(k)] = v
	}

	return nil
}

func (sa *SQLAnalyzer) analyzeSelect(stmt *sqlparser.Select) {
	sa.SQLTypes["SELECT"]++
	sa.analyzeTables(stmt.From)
	sa.analyzeFields(stmt.SelectExprs)
	sa.analyzeComplexity(stmt)
}

func (sa *SQLAnalyzer) analyzeInsert(stmt *sqlparser.Insert) {
	sa.SQLTypes["INSERT"]++
	sa.analyzeSingleTable(&sqlparser.AliasedTableExpr{
		Expr: stmt.Table,
	})
	sa.analyzeColumns(stmt.Columns)

	if rows, ok := stmt.Rows.(sqlparser.Values); ok && len(rows) > 0 {
		sa.analyzeBatchInsert(sqlparser.String(stmt))
	}
}

func (sa *SQLAnalyzer) analyzeBatchInsert(sqlString string) {
	if strings.Contains(sqlString, "/* FOREACH_START */") {
		sa.BatchInserts++

		// 提取FOREACH_START和FOREACH_END之间的内容
		start := strings.Index(sqlString, "/* FOREACH_START */")
		end := strings.Index(sqlString, "/* FOREACH_END */")
		if start != -1 && end != -1 && start < end {
			foreachContent := sqlString[start+len("/* FOREACH_START */") : end]

			// 分析foreach内部的列
			columns := extractColumns(foreachContent)
			for _, col := range columns {
				sa.Fields[col]++
			}
		}
	}
}

func extractColumns(content string) []string {
	// 使用正则表达式提取列名
	re := regexp.MustCompile(`\?`)
	matches := re.FindAllStringIndex(content, -1)

	columns := make([]string, len(matches))
	for i := range matches {
		columns[i] = fmt.Sprintf("column_%d", i+1)
	}
	return columns
}

func (sa *SQLAnalyzer) analyzeReplace(stmt *sqlparser.Insert) {
	sa.SQLTypes["REPLACE"]++
	sa.analyzeSingleTable(&sqlparser.AliasedTableExpr{
		Expr: stmt.Table,
	})
	sa.analyzeColumns(stmt.Columns)
}

func (sa *SQLAnalyzer) analyzeUpdate(stmt *sqlparser.Update) {
	sa.SQLTypes["UPDATE"]++
	sa.analyzeTables(stmt.TableExprs)
	sa.analyzeUpdateExprs(stmt.Exprs)
}

func (sa *SQLAnalyzer) analyzeDelete(stmt *sqlparser.Delete) {
	sa.SQLTypes["DELETE"]++
	sa.analyzeTables(stmt.TableExprs)
}

func (sa *SQLAnalyzer) analyzeUnion(stmt *sqlparser.Union) {
	sa.SQLTypes["UNION"]++
	sa.UnionOperations++
	//sa.analyzeSelect(stmt.Left.(*sqlparser.Select))
	//sa.analyzeSelect(stmt.Right.(*sqlparser.Select))
}

func (sa *SQLAnalyzer) analyzeSingleTable(tableExpr sqlparser.TableExpr) {
	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		if name, ok := t.Expr.(sqlparser.TableName); ok {
			sa.Tables[name.Name.String()]++
		}
	}
}

func (sa *SQLAnalyzer) analyzeColumns(columns sqlparser.Columns) {
	for _, col := range columns {
		sa.Fields[col.String()]++
	}
}

func (sa *SQLAnalyzer) analyzeUpdateExprs(exprs sqlparser.UpdateExprs) {
	for _, expr := range exprs {
		sa.Fields[expr.Name.Name.String()]++
	}
}

func (sa *SQLAnalyzer) analyzeTables(tables sqlparser.TableExprs) {
	for _, table := range tables {
		switch t := table.(type) {
		case *sqlparser.AliasedTableExpr:
			if name, ok := t.Expr.(sqlparser.TableName); ok {
				sa.Tables[name.Name.String()]++
			}
		}
	}
}

func (sa *SQLAnalyzer) analyzeFields(exprs sqlparser.SelectExprs) {
	for _, expr := range exprs {
		switch e := expr.(type) {
		case *sqlparser.AliasedExpr:
			if col, ok := e.Expr.(*sqlparser.ColName); ok {
				sa.Fields[col.Name.String()]++
			}
		}
	}
}

func (sa *SQLAnalyzer) analyzeComplexity(stmt *sqlparser.Select) {
	if stmt.Where != nil || stmt.GroupBy != nil || stmt.Having != nil ||
		stmt.OrderBy != nil || stmt.Limit != nil || len(stmt.From) > 1 ||
		sa.UnionOperations > 0 {
		sa.ComplexQueries++
	}

	if len(stmt.From) > 1 {
		sa.JoinOperations += len(stmt.From) - 1
	}

	if stmt.Where != nil {
		sa.analyzeWhere(stmt.Where)
	}

	if stmt.GroupBy != nil {
		sa.analyzeGroupBy(stmt.GroupBy)
	}

	if stmt.OrderBy != nil {
		sa.OrderByOperations++
	}

	if stmt.Limit != nil {
		sa.LimitOperations++
	}

	if stmt.Distinct != "" {
		sa.DistinctQueries++
	}

	sa.analyzeSelectExprs(stmt.SelectExprs)

	tableAliases := sa.getTableAliases(stmt.From)
	sa.analyzeWhereForIndexRecommendations(stmt.Where, tableAliases)

	sa.analyzeJoins(stmt.From)
}

func (sa *SQLAnalyzer) getTableAliases(tableExprs sqlparser.TableExprs) map[string]string {
	aliases := make(map[string]string)
	for _, expr := range tableExprs {
		switch tableExpr := expr.(type) {
		case *sqlparser.AliasedTableExpr:
			if tableName, ok := tableExpr.Expr.(sqlparser.TableName); ok {
				if tableExpr.As.String() != "" {
					aliases[tableExpr.As.String()] = tableName.Name.String()
				} else {
					aliases[tableName.Name.String()] = tableName.Name.String()
				}
			}
		case *sqlparser.JoinTableExpr:
			// 递归处理 JOIN 的两边
			for k, v := range sa.getTableAliases(sqlparser.TableExprs{tableExpr.LeftExpr}) {
				aliases[k] = v
			}
			for k, v := range sa.getTableAliases(sqlparser.TableExprs{tableExpr.RightExpr}) {
				aliases[k] = v
			}
		}
	}
	return aliases
}

func (sa *SQLAnalyzer) analyzeWhere(where *sqlparser.Where) {
	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch node := node.(type) {
		case *sqlparser.Subquery:
			sa.SubQueries++
		case *sqlparser.FuncExpr:
			sa.AggregationFuncs[node.Name.String()]++
		}
		return true, nil
	}, where.Expr)
}

func (sa *SQLAnalyzer) analyzeWhereForIndexRecommendations(where *sqlparser.Where, tableAliases map[string]string) {
	if where == nil {
		return
	}

	fieldCombinations := make(map[string]map[string]struct{})

	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch node := node.(type) {
		case *sqlparser.ComparisonExpr:
			if colName, ok := node.Left.(*sqlparser.ColName); ok {
				// 检查字段是否被忽略
				if sa.IgnoredFields[colName.Name.String()] {
					return true, nil
				}

				table := colName.Qualifier.Name.String()
				if actualTable, ok := tableAliases[table]; ok {
					table = actualTable
				} else if table == "" {
					if len(tableAliases) == 1 {
						for _, t := range tableAliases {
							table = t
							break
						}
					}
				}
				if table != "" {
					if _, ok := fieldCombinations[table]; !ok {
						fieldCombinations[table] = make(map[string]struct{})
					}
					fieldCombinations[table][colName.Name.String()] = struct{}{}
				}
			}
		}
		return true, nil
	}, where.Expr)

	for table, fields := range fieldCombinations {
		uniqueFields := make([]string, 0, len(fields))
		for field := range fields {
			uniqueFields = append(uniqueFields, field)
		}
		sort.Strings(uniqueFields)
		fieldCombination := strings.Join(uniqueFields, ",")

		if _, ok := sa.IndexRecommendations[table]; !ok {
			sa.IndexRecommendations[table] = &TableIndexRecommendation{
				Table:             table,
				FieldCombinations: make(map[string]int),
			}
		}
		sa.IndexRecommendations[table].FieldCombinations[fieldCombination]++
	}
}

func (sa *SQLAnalyzer) analyzeGroupBy(groupBy sqlparser.GroupBy) {
	for _, expr := range groupBy {
		if funcExpr, ok := expr.(*sqlparser.FuncExpr); ok {
			sa.AggregationFuncs[funcExpr.Name.String()]++
		}
	}
}

func (sa *SQLAnalyzer) analyzeSelectExprs(exprs sqlparser.SelectExprs) {
	for _, expr := range exprs {
		switch expr := expr.(type) {
		case *sqlparser.AliasedExpr:
			if funcExpr, ok := expr.Expr.(*sqlparser.FuncExpr); ok {
				sa.AggregationFuncs[funcExpr.Name.String()]++
			}
		}
	}
}

func (sa *SQLAnalyzer) analyzeJoins(tables sqlparser.TableExprs) {
	for _, table := range tables {
		switch t := table.(type) {
		case *sqlparser.JoinTableExpr:
			sa.JoinTypes[t.Join]++
		}
	}
}
