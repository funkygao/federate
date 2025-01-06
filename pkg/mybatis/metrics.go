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
			if stmt.Type == "select" {
				joinClauses := sa.extractJoinClauses(stmt.SQL)
				for _, join := range joinClauses {
					// 确保表1和表2的顺序一致
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
	}

	// 将去重后的关系转换为切片
	sa.TableRelations = make([]TableRelation, 0, len(relationMap))
	for _, relation := range relationMap {
		sa.TableRelations = append(sa.TableRelations, relation)
	}

	// 可选：按表名和连接类型排序
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
	var complexQueries []SQLComplexity
	for _, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			complexity := SQLComplexity{StatementID: stmt.ID}

			// Convert SQL to uppercase for case-insensitive matching
			upperSQL := strings.ToUpper(stmt.SQL)

			// Count various SQL elements
			joinCount := strings.Count(upperSQL, "JOIN")
			unionCount := strings.Count(upperSQL, "UNION")
			subqueryCount := strings.Count(upperSQL, "SELECT") - 1 // Subtracting 1 for the main query
			distinctCount := strings.Count(upperSQL, "DISTINCT")
			groupByCount := strings.Count(upperSQL, "GROUP BY")
			havingCount := strings.Count(upperSQL, "HAVING")
			orderByCount := strings.Count(upperSQL, "ORDER BY")
			caseCount := strings.Count(upperSQL, "CASE")

			// Calculate score
			complexity.Score += joinCount * 2
			complexity.Score += unionCount * 3
			complexity.Score += subqueryCount * 3
			complexity.Score += distinctCount * 2
			complexity.Score += groupByCount * 2
			complexity.Score += havingCount * 2
			complexity.Score += orderByCount
			complexity.Score += caseCount

			// Add reasons for complexity
			if joinCount > 0 {
				complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d JOINs", joinCount))
			}
			if unionCount > 0 {
				complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d UNIONs", unionCount))
			}
			if subqueryCount > 0 {
				complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d subqueries", subqueryCount))
			}
			if distinctCount > 0 {
				complexity.Reasons = append(complexity.Reasons, "DISTINCT")
			}
			if groupByCount > 0 {
				complexity.Reasons = append(complexity.Reasons, "GROUP BY")
			}
			if havingCount > 0 {
				complexity.Reasons = append(complexity.Reasons, "HAVING")
			}
			if orderByCount > 0 {
				complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d ORDER BY clauses", orderByCount))
			}
			if caseCount > 0 {
				complexity.Reasons = append(complexity.Reasons, fmt.Sprintf("%d CASE statements", caseCount))
			}

			// Adjust this threshold as needed
			if complexity.Score > 5 {
				complexQueries = append(complexQueries, complexity)
			}
		}
	}

	// Sort the queries by complexity score in descending order
	sort.Slice(complexQueries, func(i, j int) bool {
		return complexQueries[i].Score > complexQueries[j].Score
	})

	// Store only the top K complex queries
	if len(complexQueries) > TopK {
		sa.ComplexQueries = complexQueries[:TopK]
	} else {
		sa.ComplexQueries = complexQueries
	}
}

func (sa *SQLAnalyzer) DetectOptimisticLocking() {
	sa.OptimisticLocks = []string{}
	for _, stmts := range sa.StatementsByType {
		for _, stmt := range stmts {
			if stmt.Type == "update" {
				if strings.Contains(stmt.SQL, "version = version + 1") ||
					strings.Contains(stmt.SQL, "last_updated = NOW()") {
					sa.OptimisticLocks = append(sa.OptimisticLocks, fmt.Sprintf("Optimistic locking detected in statement: %s", stmt.ID))
				}
			}
		}
	}
}

func (sa *SQLAnalyzer) extractJoinClauses(sql string) []struct {
	LeftTable  string
	RightTable string
	Type       string
	Condition  string
} {
	stmt, err := sqlparser.Parse(sql)
	if err != nil {
		// 如果解析失败，返回空切片
		return nil
	}

	var joins []struct {
		LeftTable  string
		RightTable string
		Type       string
		Condition  string
	}

	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch n := node.(type) {
		case *sqlparser.JoinTableExpr:
			leftTable := sa.getTableName(n.LeftExpr)
			rightTable := sa.getTableName(n.RightExpr)
			joinType := strings.ToUpper(n.Join)
			condition := sqlparser.String(n.Condition.On)

			joins = append(joins, struct {
				LeftTable  string
				RightTable string
				Type       string
				Condition  string
			}{
				LeftTable:  leftTable,
				RightTable: rightTable,
				Type:       joinType,
				Condition:  condition,
			})
		}
		return true, nil
	}, stmt)

	return joins
}

func (sa *SQLAnalyzer) getTableName(tableExpr sqlparser.TableExpr) string {
	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		if tn, ok := t.Expr.(sqlparser.TableName); ok {
			return tn.Name.String()
		}
	case *sqlparser.JoinTableExpr:
		return sa.getTableName(t.LeftExpr) // 递归获取左表名
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
