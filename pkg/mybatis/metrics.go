package mybatis

import (
	"fmt"
	"sort"

	"federate/pkg/primitive"
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
	Reasons     *primitive.StringSet
}

func (sa *SQLAnalyzer) AnalyzeAll() {
	sa.AnalyzeTableUsage()
	sa.AnalyzeTableRelations()
	sa.AnalyzeSQLComplexity()
	sa.DetectOptimisticLocking()
}

func (sa *SQLAnalyzer) AnalyzeTableUsage() {
	sa.TableUsage = make(map[string]*TableUsage)
	for stmtType, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			for _, table := range stmt.Tables() {
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

	for _, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			for _, join := range stmt.ExtractJoinClauses() {
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

func (sa *SQLAnalyzer) AnalyzeSQLComplexity() {
	sa.ComplexQueries = []SQLComplexity{}

	for _, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			sa.ComplexQueries = append(sa.ComplexQueries, stmt.AnalyzeComplexity())
		}
	}

	// 按复杂度得分降序排序
	sort.Slice(sa.ComplexQueries, func(i, j int) bool {
		return sa.ComplexQueries[i].Score > sa.ComplexQueries[j].Score
	})

	// 只保留前 TopK 个复杂查询
	if len(sa.ComplexQueries) > TopK {
		sa.ComplexQueries = sa.ComplexQueries[:TopK]
	}
}

func (sa *SQLAnalyzer) DetectOptimisticLocking() {
	sa.OptimisticLocks = []*Statement{}
	for _, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			if stmt.HasOptimisticLocking() {
				sa.OptimisticLocks = append(sa.OptimisticLocks, stmt)
			}
		}
	}
}
