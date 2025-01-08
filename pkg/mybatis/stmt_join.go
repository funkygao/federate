package mybatis

import (
	"strings"

	"federate/pkg/primitive"
	"github.com/xwb1989/sqlparser"
)

type JoinClause struct {
	LeftTable  string
	RightTable string
	Type       string
}

func (s *Statement) ExtractJoinClauses() []JoinClause {
	var joins []JoinClause
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
			leftTable := s.getTableName(n.LeftExpr)
			rightTable := s.getTableName(n.RightExpr)
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

	for _, stmt := range s.PrimaryASTs() {
		extractJoins(stmt)
	}
	return joins
}

func (s *Statement) getTableName(tableExpr sqlparser.TableExpr) string {
	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		switch expr := t.Expr.(type) {
		case sqlparser.TableName:
			return expr.Name.String()
		case *sqlparser.Subquery:
			return "SUBQUERY"
		}
	case *sqlparser.JoinTableExpr:
		return s.getTableName(t.LeftExpr)
	case *sqlparser.ParenTableExpr:
		if len(t.Exprs) > 0 {
			return s.getTableName(t.Exprs[0])
		}
	}
	return ""
}

func (s *Statement) Tables() []string {
	tables := primitive.NewStringSet()
	for _, stmt := range s.PrimaryASTs() {
		sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
			switch n := node.(type) {
			case *sqlparser.AliasedTableExpr:
				tableName := s.getTableNameFromExpr(n.Expr)
				if tableName != "" {
					tables.Add(tableName)
				}
			}
			return true, nil
		}, stmt)
	}

	return tables.Values()
}

func (s *Statement) getTableNameFromExpr(expr sqlparser.SimpleTableExpr) string {
	switch t := expr.(type) {
	case sqlparser.TableName:
		return t.Name.String()
	case *sqlparser.Subquery:
		// 对于子查询，我们可以返回一个特殊的名称或者递归分析
		return "SUBQUERY"
	}
	return ""
}
