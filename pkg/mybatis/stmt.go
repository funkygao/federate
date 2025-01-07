package mybatis

import (
	"strings"

	"github.com/beevik/etree"
	"github.com/xwb1989/sqlparser"
)

type JoinClause struct {
	LeftTable  string
	RightTable string
	Type       string
}

type Statement struct {
	Filename string
	ID       string
	Type     string
	Timeout  int

	// Raw XML Node Text
	Raw string

	// 可能是多个语句组成的：SET @affected_rows = 0; UDPATE ...; set @affected_rows  = @affected_rows + row_count(); select @affected_rows as rows;
	SQL string

	SubSQL []string

	PrimarySQL []string
}

func (s *Statement) IsBatchOperation() bool {
	containsSQLOperation := func(elem *etree.Element) bool {
		text := strings.ToUpper(elem.Text())
		return strings.Contains(text, "INSERT") ||
			strings.Contains(text, "UPDATE") ||
			strings.Contains(text, "DELETE") ||
			strings.Contains(text, "SELECT")
	}

	doc := etree.NewDocument()
	if err := doc.ReadFromString(s.Raw); err != nil {
		return false
	}

	root := doc.Root()
	if root == nil {
		return false
	}

	// 检查是否有最外层的 foreach
	for _, child := range root.ChildElements() {
		if child.Tag == "foreach" && containsSQLOperation(child) {
			return true
		}
	}

	return false
}

func (s *Statement) HasOptimisticLocking() bool {
	if s.Type == "update" {
		if strings.Contains(s.SQL, "version = version + 1") {
			return true
		}
	}

	return false
}

func (s *Statement) SetPrimarySQL(sql string) {
	s.PrimarySQL = append(s.PrimarySQL, sql)
}

func (s *Statement) SubN() int {
	return len(s.SubSQL)
}

func (s *Statement) SplitSQL() []string {
	var statements []string
	var currentStmt strings.Builder
	var inString, inComment bool
	var stringDelimiter rune

	for i, char := range s.SQL {
		switch {
		case inString:
			currentStmt.WriteRune(char)
			if char == stringDelimiter && s.SQL[i-1] != '\\' {
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
		case char == '-' && i+1 < len(s.SQL) && s.SQL[i+1] == '-':
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

func (s *Statement) ExtractJoinClauses() []JoinClause {
	stmt, err := sqlparser.Parse(s.SQL)
	if err != nil {
		return nil
	}

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

	extractJoins(stmt)
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
	stmt, err := sqlparser.Parse(s.SQL)
	if err != nil {
		// 如果解析失败，返回空切片
		return []string{}
	}

	tables := make(map[string]bool)
	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch n := node.(type) {
		case *sqlparser.AliasedTableExpr:
			tableName := s.getTableNameFromExpr(n.Expr)
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
