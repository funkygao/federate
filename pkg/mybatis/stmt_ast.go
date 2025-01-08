package mybatis

import (
	"fmt"
	"log"
	"strings"

	"federate/pkg/primitive"
	"github.com/xwb1989/sqlparser"
)

var cognitiveWeights = map[string]int{
	"DISTINCT":      1,
	"GROUP BY":      1,
	"HAVING":        1,
	"ORDER BY":      1,
	"LIMIT":         1,
	"LP LIKE":       1,
	"FULL LIKE":     2,
	"CASE Expr":     1,
	"Subquery":      2,
	"OR":            1,
	"NOT":           2,
	"EXISTS":        2,
	"BETWEEN":       1,
	"CASE":          1,
	"UNION":         3,
	"UNION ALL":     1,
	"JOIN":          2,
	"MIN":           1,
	"MAX":           1,
	"AVG":           1,
	"SUM":           1,
	"COUNT":         1,
	"INSERT SELECT": 2,
	"ON DUPLICATE":  2,
}

func (s *Statement) parseSQL() (parseErrors []error) {
	for _, subSQL := range s.splitSQL() {
		stmt, err := sqlparser.Parse(subSQL)
		if err != nil {
			parseErrors = append(parseErrors, fmt.Errorf("error parsing SQL: %v. SQL: %s", err, subSQL))
			continue
		}

		sqlet := SQLet{SQL: subSQL, Stmt: stmt}

		switch stmt := stmt.(type) {
		case *sqlparser.Select:
			if !isSelectFromDual(stmt) {
				sqlet.Primary = true
				sqlet.SQLType = "SELECT"
			}
		case *sqlparser.Insert:
			sqlet.Primary = true
			sqlet.SQLType = "INSERT"
		case *sqlparser.Update:
			sqlet.Primary = true
			sqlet.SQLType = "UPDATE"
		case *sqlparser.Delete:
			sqlet.Primary = true
			sqlet.SQLType = "DELETE"
		case *sqlparser.Union:
			sqlet.Primary = true
			sqlet.SQLType = "UNION"

		case *sqlparser.Set:
			// noop

		default:
			log.Printf("Unhandled SQL type: %T\nSQL: %s", stmt, subSQL)
		}

		s.addSubSQL(sqlet)
	}

	return
}

func (s *Statement) analyzeCognitiveComplexity() {
	complexity := SQLComplexity{
		Filename:    s.Filename,
		StatementID: s.ID,
		Score:       0,
		Reasons:     primitive.NewStringSet().UseRaw(),
	}

	for _, node := range s.PrimaryASTs() {
		s.analyzeNode(node, &complexity)
	}

	// 使用权重计算最终分数
	for _, reason := range complexity.Reasons.RawValues() {
		if weight, ok := cognitiveWeights[reason]; ok {
			complexity.Score += weight
		} else {
			log.Fatalf("unknown reason for weights: %s", reason)
		}
	}

	s.Complexity = complexity
}

func (s *Statement) analyzeNode(node sqlparser.SQLNode, complexity *SQLComplexity) {
	switch n := node.(type) {
	case *sqlparser.Select:
		s.analyzeSelect(n, complexity)
	case *sqlparser.Union:
		s.analyzeUnion(n, complexity)
	case *sqlparser.Insert:
		s.analyzeInsert(n, complexity)
	case *sqlparser.Update:
		s.analyzeUpdate(n, complexity)
	case *sqlparser.Delete:
		s.analyzeDelete(n, complexity)
	case *sqlparser.Subquery:
		complexity.Reasons.Add("Subquery")
		s.analyzeNode(n.Select, complexity)
	case *sqlparser.ParenSelect: // TODO
	default:
		log.Printf("Unhandled node type: %T", n)
	}
}

func (s *Statement) analyzeSelect(selectStmt *sqlparser.Select, complexity *SQLComplexity) {
	if selectStmt.Distinct != "" {
		complexity.Reasons.Add("DISTINCT")
	}

	for _, expr := range selectStmt.SelectExprs {
		s.analyzeSelectExpr(expr, complexity)
	}

	if len(selectStmt.From) > 0 {
		s.analyzeTableExpr(selectStmt.From[0], complexity)
	}

	if selectStmt.Where != nil {
		s.analyzeExpr(selectStmt.Where.Expr, complexity)
	}

	if len(selectStmt.GroupBy) > 0 {
		complexity.Reasons.Add("GROUP BY")
	}

	if selectStmt.Having != nil {
		complexity.Reasons.Add("HAVING")
		s.analyzeExpr(selectStmt.Having.Expr, complexity)
	}

	if len(selectStmt.OrderBy) > 0 {
		complexity.Reasons.Add("ORDER BY")
	}

	if selectStmt.Limit != nil {
		complexity.Reasons.Add("LIMIT")
	}
}

func (s *Statement) analyzeSelectExpr(expr sqlparser.SelectExpr, complexity *SQLComplexity) {
	switch e := expr.(type) {
	case *sqlparser.AliasedExpr:
		s.analyzeExpr(e.Expr, complexity)
	case *sqlparser.StarExpr:
	default:
		log.Printf("Unexpected SelectExpr type: %T", e)
	}
}

// where | having
func (s *Statement) analyzeExpr(expr sqlparser.Expr, complexity *SQLComplexity) {
	switch e := expr.(type) {
	case *sqlparser.AndExpr:
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.OrExpr:
		complexity.Reasons.Add("OR")
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.ComparisonExpr:
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.Subquery:
		complexity.Reasons.Add("Subquery")
		s.analyzeNode(e.Select, complexity)
	case *sqlparser.FuncExpr:
		s.analyzeFunction(e, complexity)
	case *sqlparser.ColName:
		// 简单的列引用，不增加复杂度
	case sqlparser.ValTuple:
		for _, val := range e {
			s.analyzeExpr(val, complexity)
		}
	case *sqlparser.ParenExpr:
		s.analyzeExpr(e.Expr, complexity)
	case *sqlparser.CaseExpr:
		s.analyzeCase(e, complexity)
	case *sqlparser.NotExpr:
		complexity.Reasons.Add("NOT")
		s.analyzeExpr(e.Expr, complexity)
	case *sqlparser.ExistsExpr:
		complexity.Reasons.Add("EXISTS")
		s.analyzeNode(e.Subquery.Select, complexity)
	case *sqlparser.RangeCond:
		complexity.Reasons.Add("BETWEEN")
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.From, complexity)
		s.analyzeExpr(e.To, complexity)
	case *sqlparser.IsExpr:
		s.analyzeExpr(e.Expr, complexity)
	case sqlparser.ListArg, *sqlparser.SQLVal, *sqlparser.BinaryExpr, *sqlparser.UnaryExpr, *sqlparser.GroupConcatExpr, sqlparser.BoolVal, *sqlparser.NullVal, *sqlparser.ConvertExpr:
		// 这些是简单的值，不增加复杂度
	default:
		log.Printf("Unhandled expression type: %T", e)
	}
}

func (s *Statement) analyzeCase(caseExpr *sqlparser.CaseExpr, complexity *SQLComplexity) {
	complexity.Reasons.Add("CASE Expr")

	if caseExpr.Expr != nil {
		s.analyzeExpr(caseExpr.Expr, complexity)
	}

	for _, when := range caseExpr.Whens {
		s.analyzeExpr(when.Cond, complexity)
		s.analyzeExpr(when.Val, complexity)
	}

	if caseExpr.Else != nil {
		s.analyzeExpr(caseExpr.Else, complexity)
	}
}

func (s *Statement) analyzeUnion(union *sqlparser.Union, complexity *SQLComplexity) {
	complexity.Reasons.Add("UNION")

	s.analyzeNode(union.Left, complexity)
	s.analyzeNode(union.Right, complexity)

	if union.Type == sqlparser.UnionAllStr {
		complexity.Reasons.Add("UNION ALL")
	}
}

func (s *Statement) analyzeTableExpr(expr sqlparser.TableExpr, complexity *SQLComplexity) {
	switch t := expr.(type) {
	case *sqlparser.AliasedTableExpr:
		switch tableExpr := t.Expr.(type) {
		case sqlparser.TableName:
			// 简单的表引用
		case *sqlparser.Subquery:
			complexity.Reasons.Add("Subquery")
			s.analyzeNode(tableExpr.Select, complexity)
		}
	case *sqlparser.JoinTableExpr:
		complexity.Reasons.Add("JOIN")
		s.analyzeTableExpr(t.LeftExpr, complexity)
		s.analyzeTableExpr(t.RightExpr, complexity)
	}
}

func (s *Statement) analyzeFunction(funcExpr *sqlparser.FuncExpr, complexity *SQLComplexity) {
	switch strings.ToUpper(funcExpr.Name.String()) {
	case "MIN", "MAX", "AVG", "SUM", "COUNT":
		complexity.Reasons.Add(strings.ToUpper(funcExpr.Name.String()))
	}
}

func (s *Statement) analyzeInsert(insert *sqlparser.Insert, complexity *SQLComplexity) {
	// 分析 VALUES 子句或 SELECT 子句
	switch rows := insert.Rows.(type) {
	case *sqlparser.Select:
		complexity.Reasons.Add("INSERT SELECT")
		s.analyzeNode(rows, complexity)
	case sqlparser.Values:
	}

	// 分析 ON DUPLICATE KEY UPDATE 子句
	if insert.OnDup != nil {
		complexity.Reasons.Add("ON DUPLICATE")
		for _, expr := range insert.OnDup {
			s.analyzeExpr(expr.Expr, complexity)
		}
	}
}

func (s *Statement) analyzeUpdate(update *sqlparser.Update, complexity *SQLComplexity) {
	// 分析更新的表
	for _, tableExpr := range update.TableExprs {
		s.analyzeTableExpr(tableExpr, complexity)
	}

	// 分析 SET 子句
	for _, expr := range update.Exprs {
		s.analyzeExpr(expr.Expr, complexity)
	}

	// 分析 WHERE 子句
	if update.Where != nil {
		s.analyzeExpr(update.Where.Expr, complexity)
	}

	// 分析 ORDER BY 子句
	if len(update.OrderBy) > 0 {
		complexity.Reasons.Add("ORDER BY")
	}
}

func (s *Statement) analyzeDelete(delete *sqlparser.Delete, complexity *SQLComplexity) {
	// 分析删除的表
	for _, tableExpr := range delete.TableExprs {
		s.analyzeTableExpr(tableExpr, complexity)
	}

	// 分析 WHERE 子句
	if delete.Where != nil {
		s.analyzeExpr(delete.Where.Expr, complexity)
	}

	// 分析 ORDER BY 子句
	if len(delete.OrderBy) > 0 {
		complexity.Reasons.Add("ORDER BY")
	}
}

func isSelectFromDual(stmt *sqlparser.Select) bool {
	if len(stmt.From) == 1 {
		if tableExpr, ok := stmt.From[0].(*sqlparser.AliasedTableExpr); ok {
			if tableName, ok := tableExpr.Expr.(sqlparser.TableName); ok {
				return tableName.Name.String() == "dual"
			}
		}
	}
	return false
}
