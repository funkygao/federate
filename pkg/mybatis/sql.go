package mybatis

import (
	"fmt"
	"log"
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

	StatementsByType map[string][]*Statement

	SQLTypes           map[string]int
	Tables             map[string]int
	Fields             map[string]int
	ComplexQueries     int
	UnionOperations    int
	SubQueries         int
	AggregationFuncs   map[string]int // count, min, max, etc
	DistinctQueries    int
	OrderByOperations  int
	LimitOperations    int
	LimitWithOffset    int
	LimitWithoutOffset int
	JoinOperations     int
	JoinTypes          map[string]int
	JoinTableCounts    map[int]int
	ParsedOK           int
	TimeoutStatements  map[string]int

	IndexRecommendations map[string]*TableIndexRecommendation

	UnknownFragments map[string][]SqlFragmentRef
	UnparsableSQL    []UnparsableSQL
}

func NewSQLAnalyzer(ignoredFields []string) *SQLAnalyzer {
	sa := &SQLAnalyzer{
		IgnoredFields:     make(map[string]bool),
		SQLTypes:          make(map[string]int),
		Tables:            make(map[string]int),
		Fields:            make(map[string]int),
		AggregationFuncs:  make(map[string]int),
		JoinTypes:         make(map[string]int),
		JoinTableCounts:   make(map[int]int),
		TimeoutStatements: make(map[string]int),
		StatementsByType:  make(map[string][]*Statement),

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
	if s.Timeout > 0 {
		timeoutKey := fmt.Sprintf("%ds", s.Timeout)
		sa.TimeoutStatements[timeoutKey]++
	}

	var parseErrors []error
	for _, sqlStmt := range sa.splitSQLStatements(s.SQL) {
		sqlStmt = strings.TrimSpace(sqlStmt)
		if sqlStmt == "" {
			continue
		}

		stmt, err := sqlparser.Parse(sqlStmt)
		if err != nil {
			parseErrors = append(parseErrors, fmt.Errorf("error parsing SQL: %v. SQL: %s", err, sqlStmt))
			continue
		}

		sa.ParsedOK++
		sa.StatementsByType[s.Type] = append(sa.StatementsByType[s.Type], &s)

		switch stmt := stmt.(type) {
		case *sqlparser.Select:
			sa.analyzeSelect(stmt, s)
		case *sqlparser.Insert:
			if stmt.Action == sqlparser.ReplaceStr {
				sa.analyzeReplace(stmt, s)
			} else {
				sa.analyzeInsert(stmt, s)
			}
		case *sqlparser.Update:
			sa.analyzeUpdate(stmt, s)
		case *sqlparser.Delete:
			sa.analyzeDelete(stmt, s)
		case *sqlparser.Union:
			sa.analyzeUnion(stmt, s)
		case *sqlparser.Set:
			sa.analyzeSet(stmt)
		default:
			log.Printf("Unhandled SQL type: %T\nSQL: %s", stmt, sqlStmt)
		}
	}

	// Unify aggregation function names to uppercase
	for k, v := range sa.AggregationFuncs {
		delete(sa.AggregationFuncs, k)
		sa.AggregationFuncs[strings.ToUpper(k)] = v
	}

	if len(parseErrors) > 0 {
		return fmt.Errorf("encountered %d parse errors: %v", len(parseErrors), parseErrors)
	}

	return nil
}

func (sa *SQLAnalyzer) splitSQLStatements(sql string) []string {
	var statements []string
	var currentStmt strings.Builder
	var inString, inComment bool
	var stringDelimiter rune

	for i, char := range sql {
		switch {
		case inString:
			currentStmt.WriteRune(char)
			if char == stringDelimiter && sql[i-1] != '\\' {
				inString = false
			}
		case inComment:
			if char == '\n' {
				inComment = false
			}
			currentStmt.WriteRune(char)
		case char == '\'' || char == '"':
			inString = true
			stringDelimiter = char
			currentStmt.WriteRune(char)
		case char == '-' && i+1 < len(sql) && sql[i+1] == '-':
			inComment = true
			currentStmt.WriteRune(char)
		case char == ';' && !inString && !inComment:
			stmt := strings.TrimSpace(currentStmt.String())
			if stmt != "" {
				statements = append(statements, stmt)
			}
			currentStmt.Reset()
		default:
			currentStmt.WriteRune(char)
		}
	}

	if currentStmt.Len() > 0 {
		stmt := strings.TrimSpace(currentStmt.String())
		if stmt != "" {
			statements = append(statements, stmt)
		}
	}

	return statements
}

func (sa *SQLAnalyzer) analyzeSelect(stmt *sqlparser.Select, s Statement) {
	sa.SQLTypes["SELECT"]++
	sa.analyzeTables(stmt.From, s)
	sa.analyzeFields(stmt.SelectExprs)
	sa.analyzeComplexity(stmt)
}

func (sa *SQLAnalyzer) analyzeInsert(stmt *sqlparser.Insert, s Statement) {
	sa.SQLTypes["INSERT"]++
	sa.analyzeTable(&sqlparser.AliasedTableExpr{
		Expr: stmt.Table,
	}, s)
	sa.analyzeColumns(stmt.Columns)
}

func (sa *SQLAnalyzer) analyzeReplace(stmt *sqlparser.Insert, s Statement) {
	sa.SQLTypes["REPLACE"]++
	sa.analyzeTable(&sqlparser.AliasedTableExpr{
		Expr: stmt.Table,
	}, s)
	sa.analyzeColumns(stmt.Columns)
}

func (sa *SQLAnalyzer) analyzeUpdate(stmt *sqlparser.Update, s Statement) {
	sa.SQLTypes["UPDATE"]++
	sa.analyzeTables(stmt.TableExprs, s)
	sa.analyzeUpdateExprs(stmt.Exprs)
}

func (sa *SQLAnalyzer) analyzeDelete(stmt *sqlparser.Delete, s Statement) {
	sa.SQLTypes["DELETE"]++
	sa.analyzeTables(stmt.TableExprs, s)
}

func (sa *SQLAnalyzer) analyzeUnion(stmt *sqlparser.Union, s Statement) {
	sa.SQLTypes["UNION"]++
	sa.UnionOperations++
	//sa.analyzeSelect(stmt.Left.(*sqlparser.Select))
	//sa.analyzeSelect(stmt.Right.(*sqlparser.Select))
}

func (sa *SQLAnalyzer) analyzeSet(stmt *sqlparser.Set) {
	sa.SQLTypes["SET @"]++
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

func (sa *SQLAnalyzer) analyzeTables(tables sqlparser.TableExprs, s Statement) {
	for _, table := range tables {
		sa.analyzeTable(table, s)
	}
}

func (sa *SQLAnalyzer) analyzeTable(tableExpr sqlparser.TableExpr, s Statement) {
	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		if name, ok := t.Expr.(sqlparser.TableName); ok {
			tableName := name.Name.String()

			if tableName == "dual" {
				// e,g. select @affected_rows as rows，sqlparser 会自动引入 "dual" 表
				if !strings.Contains(strings.ToLower(s.SQL), "dual") {
					return
				}
			}

			sa.Tables[tableName]++
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
	joinCount, joinTypes, joinTableCount := sa.analyzeJoins(stmt.From)
	// 只有在实际有 JOIN 操作时才更新相关统计
	if joinCount > 0 {
		sa.JoinOperations += joinCount
		for joinType, count := range joinTypes {
			sa.JoinTypes[joinType] += count
		}
		sa.JoinTableCounts[joinTableCount]++
	}

	if stmt.Where != nil || stmt.GroupBy != nil || stmt.Having != nil ||
		stmt.OrderBy != nil || stmt.Limit != nil || joinCount > 0 ||
		sa.UnionOperations > 0 {
		sa.ComplexQueries++
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
		if stmt.Limit.Offset != nil {
			sa.LimitWithOffset++
		} else {
			sa.LimitWithoutOffset++
		}
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
	if where == nil {
		return
	}

	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch node := node.(type) {
		case *sqlparser.Subquery:
			sa.SubQueries++
		case *sqlparser.ComparisonExpr:
			if _, ok := node.Right.(*sqlparser.Subquery); ok {
				sa.SubQueries++
			}
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

func (sa *SQLAnalyzer) countJoins(tables sqlparser.TableExprs) int {
	count := 0
	for _, table := range tables {
		switch t := table.(type) {
		case *sqlparser.JoinTableExpr:
			count++
			count += sa.countJoins(sqlparser.TableExprs{t.LeftExpr})
			count += sa.countJoins(sqlparser.TableExprs{t.RightExpr})
		}
	}
	return count
}

func (sa *SQLAnalyzer) analyzeJoins(tables sqlparser.TableExprs) (int, map[string]int, int) {
	joinCount := 0
	joinTypes := make(map[string]int)
	tableCount := 0

	var analyzeTable func(sqlparser.TableExpr) int
	analyzeTable = func(table sqlparser.TableExpr) int {
		switch t := table.(type) {
		case *sqlparser.JoinTableExpr:
			joinType := strings.ToUpper(t.Join)
			joinTypes[joinType]++
			joinCount++
			leftCount := analyzeTable(t.LeftExpr)
			rightCount := analyzeTable(t.RightExpr)
			return leftCount + rightCount
		case *sqlparser.AliasedTableExpr:
			tableCount++
			return 1
		case *sqlparser.ParenTableExpr:
			subCount := 0
			for _, expr := range t.Exprs {
				subCount += analyzeTable(expr)
			}
			return subCount
		}
		return 0
	}

	totalTables := 0
	for _, table := range tables {
		totalTables += analyzeTable(table)
	}

	// 只有当实际发生 JOIN 时才记录表数量
	if joinCount > 0 {
		sa.JoinTableCounts[totalTables]++
	}

	return joinCount, joinTypes, tableCount
}
