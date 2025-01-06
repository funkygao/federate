package mybatis

import (
	"fmt"
	"sort"
	"strings"

	"github.com/xwb1989/sqlparser"
)

type TableUsage struct {
	Name     string
	UseCount int
	InSelect int
	InInsert int
	InUpdate int
	InDelete int
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
	Reasons     []string
}

func (sa *SQLAnalyzer) AnalyzeAll() {
	sa.AnalyzeTableUsage()
	sa.AnalyzeTableRelations()
	sa.DetectPerformanceBottlenecks()
	sa.AnalyzeSQLComplexity()
	sa.DetectOptimisticLocking()
}

func (sa *SQLAnalyzer) AnalyzeTableUsage() {
	sa.TableUsage = make(map[string]*TableUsage)
	for stmtType, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			tables := sa.getTablesFromSQL(stmt.SQL)
			for _, table := range tables {
				if table == "SUBQUERY" {
					continue // 跳过子查询
				}
				if _, ok := sa.TableUsage[table]; !ok {
					sa.TableUsage[table] = &TableUsage{Name: table}
				}
				sa.TableUsage[table].UseCount++
				switch stmtType {
				case "select":
					sa.TableUsage[table].InSelect++
				case "insert":
					sa.TableUsage[table].InInsert++
				case "update":
					sa.TableUsage[table].InUpdate++
				case "delete":
					sa.TableUsage[table].InDelete++
				}
			}
		}
	}
}

func (sa *SQLAnalyzer) AnalyzeTableRelations() {
	relationMap := make(map[string]TableRelation)

	for _, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			joinClauses := sa.extractJoinClauses(stmt.SQL)
			for _, join := range joinClauses {
				// Skip if either table is empty or "SUBQUERY"
				if join.LeftTable == "" || join.RightTable == "" ||
					join.LeftTable == "SUBQUERY" || join.RightTable == "SUBQUERY" {
					continue
				}

				// Ensure table1 and table2 order is consistent
				table1, table2 := join.LeftTable, join.RightTable
				if table1 > table2 {
					table1, table2 = table2, table1
				}

				key := fmt.Sprintf("%s:%s:%s", table1, table2, join.Type)
				relationMap[key] = TableRelation{
					Table1:   table1,
					Table2:   table2,
					JoinType: join.Type,
				}
			}
		}
	}

	// Convert the deduplicated relations to a slice
	sa.TableRelations = make([]TableRelation, 0, len(relationMap))
	for _, relation := range relationMap {
		sa.TableRelations = append(sa.TableRelations, relation)
	}

	// Sort by table names and join type
	sort.Slice(sa.TableRelations, func(i, j int) bool {
		if sa.TableRelations[i].Table1 != sa.TableRelations[j].Table1 {
			return sa.TableRelations[i].Table1 < sa.TableRelations[j].Table1
		}
		if sa.TableRelations[i].Table2 != sa.TableRelations[j].Table2 {
			return sa.TableRelations[i].Table2 < sa.TableRelations[j].Table2
		}
		return sa.TableRelations[i].JoinType < sa.TableRelations[j].JoinType
	})
}

func (sa *SQLAnalyzer) DetectPerformanceBottlenecks() {
	sa.PerformanceBottlenecks = []string{}
}

func (sa *SQLAnalyzer) AnalyzeSQLComplexity() {
	complexityMap := make(map[string]SQLComplexity)

	for _, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			complexity := SQLComplexity{
				StatementID: stmt.ID,
				Filename:    stmt.Filename,
			}

			// Use sa.splitSQLStatements to split the SQL
			sqlStatements := sa.splitSQLStatements(stmt.SQL)

			totalScore := 0
			reasons := make(map[string]int)

			for _, sqlStmt := range sqlStatements {
				upperSQL := strings.ToUpper(sqlStmt)

				// Count various SQL elements
				joinCount := strings.Count(upperSQL, "JOIN")
				unionCount := strings.Count(upperSQL, "UNION")
				subqueryCount := strings.Count(upperSQL, "SELECT") - 1 // Subtracting 1 for the main query
				distinctCount := strings.Count(upperSQL, "DISTINCT")
				groupByCount := strings.Count(upperSQL, "GROUP BY")
				havingCount := strings.Count(upperSQL, "HAVING")
				orderByCount := strings.Count(upperSQL, "ORDER BY")
				insertCount := strings.Count(upperSQL, "INSERT")

				caseCount := 0
				if strings.Contains(upperSQL, "CASE") {
					// Only count CASE statements in WHERE clauses or JOIN conditions
					caseInWhereOrJoin := strings.Contains(upperSQL, "WHERE CASE") ||
						strings.Contains(upperSQL, "ON CASE") ||
						strings.Contains(upperSQL, "AND CASE") ||
						strings.Contains(upperSQL, "OR CASE")
					if caseInWhereOrJoin {
						caseCount = strings.Count(upperSQL, "CASE")
					}
				}

				// Special check for INSERT INTO SELECT
				insertIntoSelectCount := 0
				if strings.Contains(upperSQL, "INSERT") && strings.Contains(upperSQL, "SELECT") {
					insertIntoSelectCount = 1
				}

				// Calculate score
				score := joinCount*2 + unionCount*3 + subqueryCount*3 + distinctCount*2 +
					groupByCount*2 + havingCount*2 + orderByCount + caseCount +
					insertCount + insertIntoSelectCount*5

				totalScore += score

				// Add reasons
				if joinCount > 0 {
					reasons["JOINs"] += joinCount
				}
				if unionCount > 0 {
					reasons["UNIONs"] += unionCount
				}
				if subqueryCount > 0 {
					reasons["subqueries"] += subqueryCount
				}
				if distinctCount > 0 {
					reasons["DISTINCT"] += distinctCount
				}
				if groupByCount > 0 {
					reasons["GROUP BY"] += groupByCount
				}
				if havingCount > 0 {
					reasons["HAVING"] += havingCount
				}
				if orderByCount > 0 {
					reasons["ORDER BY"] += orderByCount
				}
				if caseCount > 0 {
					reasons["CASE statements in conditions"] = caseCount
				}
				if insertIntoSelectCount > 0 {
					reasons["INSERT INTO SELECT"] += insertIntoSelectCount
				} else if insertCount > 0 {
					reasons["INSERT statements"] += insertCount
				}
			}

			complexity.Score = totalScore

			// Convert reasons map to slice of strings
			for reason, count := range reasons {
				if count > 1 {
					complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d %s", count, reason))
				} else {
					complexity.Reasons = append(complexity.Reasons, reason)
				}
			}

			// Use filename and statement ID as key to avoid duplicates
			key := fmt.Sprintf("%s:%s", stmt.Filename, stmt.ID)
			complexityMap[key] = complexity
		}
	}

	// Convert map to slice
	sa.ComplexQueries = make([]SQLComplexity, 0, len(complexityMap))
	for _, complexity := range complexityMap {
		sa.ComplexQueries = append(sa.ComplexQueries, complexity)
	}

	// Sort by complexity score in descending order
	sort.Slice(sa.ComplexQueries, func(i, j int) bool {
		return sa.ComplexQueries[i].Score > sa.ComplexQueries[j].Score
	})

	// Keep only top K complex queries
	if len(sa.ComplexQueries) > TopK {
		sa.ComplexQueries = sa.ComplexQueries[:TopK]
	}
}

func (sa *SQLAnalyzer) DetectOptimisticLocking() {
	sa.OptimisticLocks = []*Statement{}
	for _, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			if stmt.Type == "update" {
				if strings.Contains(stmt.SQL, "version = version + 1") ||
					strings.Contains(stmt.SQL, "last_updated = NOW()") {
					sa.OptimisticLocks = append(sa.OptimisticLocks, stmt)
				}
			}
		}
	}
}

func (sa *SQLAnalyzer) extractJoinClauses(sql string) []struct {
	LeftTable  string
	RightTable string
	Type       string
} {
	stmt, err := sqlparser.Parse(sql)
	if err != nil {
		return nil
	}

	var joins []struct {
		LeftTable  string
		RightTable string
		Type       string
	}

	var extractJoins func(node sqlparser.SQLNode)
	extractJoins = func(node sqlparser.SQLNode) {
		switch n := node.(type) {
		case *sqlparser.Select:
			// Handle main FROM clause
			for _, tableExpr := range n.From {
				extractJoins(tableExpr)
			}
			// Handle subqueries in the SELECT list
			for _, expr := range n.SelectExprs {
				switch e := expr.(type) {
				case *sqlparser.AliasedExpr:
					if subquery, ok := e.Expr.(*sqlparser.Subquery); ok {
						extractJoins(subquery.Select)
					}
				}
			}
			// Handle subqueries in WHERE clause
			if n.Where != nil {
				extractJoins(n.Where.Expr)
			}
		case *sqlparser.JoinTableExpr:
			leftTable := sa.getTableName(n.LeftExpr)
			rightTable := sa.getTableName(n.RightExpr)
			joinType := strings.ToUpper(n.Join)
			joins = append(joins, struct {
				LeftTable  string
				RightTable string
				Type       string
			}{
				LeftTable:  leftTable,
				RightTable: rightTable,
				Type:       joinType,
			})
			extractJoins(n.LeftExpr)
			extractJoins(n.RightExpr)
		case *sqlparser.AliasedTableExpr:
			if subquery, ok := n.Expr.(*sqlparser.Subquery); ok {
				extractJoins(subquery.Select)
			}
		case *sqlparser.Where:
			sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
				extractJoins(node)
				return true, nil
			}, n.Expr)
		case *sqlparser.Subquery:
			extractJoins(n.Select)
		}
	}

	extractJoins(stmt)
	return joins
}

func (sa *SQLAnalyzer) getTableName(tableExpr sqlparser.TableExpr) string {
	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		switch expr := t.Expr.(type) {
		case sqlparser.TableName:
			return expr.Name.String()
		case *sqlparser.Subquery:
			return "SUBQUERY"
		}
	case *sqlparser.JoinTableExpr:
		return sa.getTableName(t.LeftExpr)
	case *sqlparser.ParenTableExpr:
		if len(t.Exprs) > 0 {
			return sa.getTableName(t.Exprs[0])
		}
	}
	return ""
}

func (sa *SQLAnalyzer) getTablesFromSQL(sql string) []string {
	stmt, err := sqlparser.Parse(sql)
	if err != nil {
		// 如果解析失败，返回空切片
		return []string{}
	}

	tables := make(map[string]bool)
	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch n := node.(type) {
		case *sqlparser.AliasedTableExpr:
			tableName := sa.getTableNameFromExpr(n.Expr)
			if tableName != "" {
				tables[tableName] = true
			}
		}
		return true, nil
	}, stmt)

	result := make([]string, 0, len(tables))
	for table := range tables {
		result = append(result, table)
	}
	return result
}

func (sa *SQLAnalyzer) getTableNameFromExpr(expr sqlparser.SimpleTableExpr) string {
	switch t := expr.(type) {
	case sqlparser.TableName:
		return t.Name.String()
	case *sqlparser.Subquery:
		// 对于子查询，我们可以返回一个特殊的名称或者递归分析
		return "SUBQUERY"
	}
	return ""
}
