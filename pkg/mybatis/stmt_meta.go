package mybatis

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/xwb1989/sqlparser"
)

type StatementMetadata struct {
	SQLTypes             []string
	TableAliases         map[string]string // 别名 -> 真实表名
	Tables               []string
	Fields               []string
	AggregationFuncs     map[string]map[string]int // SQLType -> FuncName -> Count
	IndexRecommendations map[string]*TableIndexRecommendation
	GroupByFields        map[string]int
	OrderByFields        map[string]int
	JoinOperations       int
	JoinTypes            map[string]int
	JoinTableCount       int
	JoinConditions       map[string]int
	IndexHints           map[string]int
	UnionOperations      int
	SubQueries           int
	HasDistinct          bool
	HasOrderBy           bool
	HasGroupBy           bool
	HasHaving            bool
	HasLimit             bool
	HasOffset            bool
}

func (s *Statement) analyzeMetadata() {
	s.Metadata = StatementMetadata{
		Tables:               []string{},
		Fields:               []string{},
		SQLTypes:             []string{},
		TableAliases:         make(map[string]string),
		AggregationFuncs:     make(map[string]map[string]int),
		IndexRecommendations: make(map[string]*TableIndexRecommendation),
		JoinTypes:            make(map[string]int),
		JoinConditions:       make(map[string]int),
		IndexHints:           make(map[string]int),
	}

	for _, stmt := range s.PrimaryASTs() {
		s.Metadata.analyzeStmt(stmt)
	}
}

func (s *StatementMetadata) analyzeStmt(stmt sqlparser.Statement) {
	switch v := stmt.(type) {
	case *sqlparser.Select:
		if !isSelectFromDual(v) {
			s.addSQLType("SELECT")
			s.analyzeSelect(v)
		}
	case *sqlparser.Insert:
		s.addSQLType("INSERT")
		s.analyzeInsert(v)
	case *sqlparser.Update:
		s.addSQLType("UPDATE")
		s.analyzeUpdate(v)
	case *sqlparser.Delete:
		s.addSQLType("DELETE")
		s.analyzeDelete(v)
	case *sqlparser.Union:
		s.addSQLType("UNION")
		s.analyzeUnion(v)
	case *sqlparser.Set:
		s.addSQLType("Set @")
	default:
		log.Printf("unknown stmt type: %T", v)
	}
}

func (s *StatementMetadata) addSQLType(sqlType string) {
	s.SQLTypes = append(s.SQLTypes, sqlType)
}

func (s *StatementMetadata) ContainsSelect() bool {
	for _, s := range s.SQLTypes {
		if s == "SELECT" {
			return true
		}
	}
	return false
}

func (s *StatementMetadata) analyzeSelect(stmt *sqlparser.Select) {
	sqlType := "SELECT"
	s.analyzeTables(stmt.From)
	s.analyzeExpressions(stmt.SelectExprs, sqlType)
	s.analyzeWhere(stmt.Where, sqlType)
	s.analyzeGroupBy(stmt.GroupBy, sqlType)
	s.analyzeHaving(stmt.Having, sqlType)
	s.analyzeOrderBy(stmt.OrderBy, sqlType)
	s.analyzeLimit(stmt.Limit)

	if stmt.Distinct != "" {
		s.HasDistinct = true
	}

	if len(stmt.OrderBy) > 0 {
		s.HasOrderBy = true
	}

	if len(stmt.GroupBy) > 0 {
		s.HasGroupBy = true
	}
}

func (s *StatementMetadata) analyzeInsert(stmt *sqlparser.Insert) {
	s.Tables = append(s.Tables, stmt.Table.Name.String())
	s.analyzeColumns(stmt.Columns)
	if selectStmt, ok := stmt.Rows.(*sqlparser.Select); ok {
		s.analyzeSelect(selectStmt)
	}
}

func (s *StatementMetadata) analyzeUpdate(stmt *sqlparser.Update) {
	sqlType := "UPDATE"
	s.analyzeTables(stmt.TableExprs)
	s.analyzeUpdateExpressions(stmt.Exprs, sqlType)
	s.analyzeWhere(stmt.Where, sqlType)
	s.analyzeOrderBy(stmt.OrderBy, sqlType)
	s.analyzeLimit(stmt.Limit)
}

func (s *StatementMetadata) analyzeDelete(stmt *sqlparser.Delete) {
	sqlType := "UPDATE"
	s.analyzeTables(stmt.TableExprs)
	s.analyzeWhere(stmt.Where, sqlType)
	s.analyzeOrderBy(stmt.OrderBy, sqlType)
	s.analyzeLimit(stmt.Limit)
}

func (s *StatementMetadata) analyzeUnion(stmt *sqlparser.Union) {
	s.UnionOperations++
	s.analyzeStmt(stmt.Left)
	s.analyzeStmt(stmt.Right)
}

func (s *StatementMetadata) analyzeTables(tables sqlparser.TableExprs) {
	for _, table := range tables {
		switch t := table.(type) {
		case *sqlparser.AliasedTableExpr:
			if name, ok := t.Expr.(sqlparser.TableName); ok {
				realName := name.Name.String()
				s.Tables = append(s.Tables, realName)
				if t.As.String() != "" {
					s.TableAliases[t.As.String()] = realName
				} else {
					s.TableAliases[realName] = realName // 如果没有别名，使用真实名称作为键
				}
			}

			// 捕获索引提示
			if t.Hints != nil {
				hintType := strings.ToUpper(t.Hints.Type)
				for _, index := range t.Hints.Indexes {
					indexName := index.String()
					hintKey := fmt.Sprintf("%s:%s", hintType, indexName)
					if s.IndexHints == nil {
						s.IndexHints = make(map[string]int)
					}
					s.IndexHints[hintKey]++
				}
			}
		case *sqlparser.JoinTableExpr:
			s.JoinOperations++
			s.JoinTypes[t.Join]++
			s.analyzeTables(sqlparser.TableExprs{t.LeftExpr, t.RightExpr})
		}
	}
}

func (s *StatementMetadata) analyzeExpressions(exprs sqlparser.SelectExprs, sqlType string) {
	for _, expr := range exprs {
		switch e := expr.(type) {
		case *sqlparser.AliasedExpr:
			s.analyzeExpr(e.Expr, sqlType)
		case *sqlparser.StarExpr:
			// Handle star expressions
		}
	}
}

func (s *StatementMetadata) addAggregationFunc(sqlType, funcName string) {
	if s.AggregationFuncs[sqlType] == nil {
		s.AggregationFuncs[sqlType] = make(map[string]int)
	}
	s.AggregationFuncs[sqlType][strings.ToUpper(funcName)]++
}

func (s *StatementMetadata) analyzeExpr(expr sqlparser.Expr, context string) {
	switch e := expr.(type) {
	case *sqlparser.ColName:
		table := e.Qualifier.Name.String()
		field := e.Name.String()
		s.Fields = append(s.Fields, e.Name.String())
		s.updateIndexRecommendation(table, field, context)

	case *sqlparser.FuncExpr:
		s.addAggregationFunc(context, e.Name.String())
		for _, arg := range e.Exprs {
			switch a := arg.(type) {
			case *sqlparser.AliasedExpr:
				s.analyzeExpr(a.Expr, context)
			case *sqlparser.StarExpr:
				// 处理 * 表达式，可能不需要特殊处理
			default:
				log.Printf("Unexpected expression type in function argument: %T", arg)
			}
		}

	case *sqlparser.GroupConcatExpr:
		s.addAggregationFunc(context, "GROUP_CONCAT")
		for _, expr := range e.Exprs {
			if aliasedExpr, ok := expr.(*sqlparser.AliasedExpr); ok {
				s.analyzeExpr(aliasedExpr.Expr, context)
			}
		}
		if e.OrderBy != nil {
			for _, order := range e.OrderBy {
				s.analyzeExpr(order.Expr, context)
			}
		}
		if e.Separator != "" {
			// 直接使用分隔符字符串，不需要进一步分析
		}

	case *sqlparser.Subquery:
		s.SubQueries++
		s.analyzeStmt(e.Select)
	case *sqlparser.BinaryExpr:
		s.analyzeExpr(e.Left, context)
		s.analyzeExpr(e.Right, context)
	case *sqlparser.UnaryExpr:
		s.analyzeExpr(e.Expr, context)
	case *sqlparser.ParenExpr:
		s.analyzeExpr(e.Expr, context)
	case *sqlparser.ConvertExpr:
		s.analyzeExpr(e.Expr, context)
	case *sqlparser.CaseExpr:
		if e.Expr != nil {
			s.analyzeExpr(e.Expr, context)
		}
		for _, when := range e.Whens {
			s.analyzeExpr(when.Cond, context)
			s.analyzeExpr(when.Val, context)
		}
		if e.Else != nil {
			s.analyzeExpr(e.Else, context)
		}
	case *sqlparser.AndExpr:
		s.analyzeExpr(e.Left, context)
		s.analyzeExpr(e.Right, context)
	case *sqlparser.OrExpr:
		s.analyzeExpr(e.Left, context)
		s.analyzeExpr(e.Right, context)
	case *sqlparser.NotExpr:
		s.analyzeExpr(e.Expr, context)
	case *sqlparser.ComparisonExpr:
		s.analyzeExpr(e.Left, context)
		s.analyzeExpr(e.Right, context)
	case *sqlparser.IsExpr:
		s.analyzeExpr(e.Expr, context)
	case *sqlparser.ExistsExpr:
		s.analyzeExpr(e.Subquery, context)
	case sqlparser.ValTuple:
		for _, val := range e {
			s.analyzeExpr(val, context)
		}
	case *sqlparser.RangeCond:
		s.analyzeExpr(e.Left, context)
		s.analyzeExpr(e.From, context)
		s.analyzeExpr(e.To, context)
	case *sqlparser.SQLVal, *sqlparser.IntervalExpr, sqlparser.BoolVal, *sqlparser.NullVal:
	default:
		// 对于其他类型的表达式，我们可能不需要进一步分析
		// 但是可以记录日志以便于调试
		log.Printf("Unhandled expression type: %T", expr)
	}
}

func (s *StatementMetadata) updateIndexRecommendation(table, field, context string) {
	realTable, ok := s.TableAliases[table]
	if !ok {
		// 如果找不到别名，就使用原始表名（可能已经是真实表名）
		realTable = table
	}

	table = realTable

	if s.IndexRecommendations[table] == nil {
		s.IndexRecommendations[table] = &TableIndexRecommendation{
			Table:             table,
			FieldCombinations: make(map[string]int),
			JoinFields:        make(map[string]int),
			WhereFields:       make(map[string]int),
			GroupByFields:     make(map[string]int),
			OrderByFields:     make(map[string]int),
		}
	}

	rec := s.IndexRecommendations[table]
	switch context {
	case "WHERE":
		rec.WhereFields[field]++
	case "JOIN":
		rec.JoinFields[field]++
	case "GROUP BY":
		rec.GroupByFields[field]++
	case "ORDER BY":
		rec.OrderByFields[field]++
	}

	// 更新字段组合
	fields := []string{}
	for f := range rec.WhereFields {
		fields = append(fields, f)
	}
	for f := range rec.JoinFields {
		fields = append(fields, f)
	}
	sort.Strings(fields)
	combination := strings.Join(fields, ",")
	rec.FieldCombinations[combination]++
}

func (s *StatementMetadata) analyzeWhere(where *sqlparser.Where, sqlType string) {
	if where == nil {
		return
	}
	s.analyzeExpr(where.Expr, "WHERE")
}

func (s *StatementMetadata) analyzeGroupBy(groupBy sqlparser.GroupBy, sqlType string) {
	if len(groupBy) > 0 {
		if s.GroupByFields == nil {
			s.GroupByFields = make(map[string]int)
		}

		var fields []string
		for _, expr := range groupBy {
			field := s.getFieldName(expr)
			fields = append(fields, field)

			s.analyzeExpr(expr, "GROUP BY")
		}

		combination := strings.Join(fields, ", ")
		s.GroupByFields[combination]++
	}
}

func (s *StatementMetadata) getFieldName(expr sqlparser.Expr) string {
	switch e := expr.(type) {
	case *sqlparser.ColName:
		return e.Name.String()
	case *sqlparser.FuncExpr:
		return e.Name.String() // 返回函数名
	case *sqlparser.ParenExpr:
		return "(" + s.getFieldName(e.Expr) + ")"
	case *sqlparser.BinaryExpr:
		return s.getFieldName(e.Left) + " " + e.Operator + " " + s.getFieldName(e.Right)
	case *sqlparser.UnaryExpr:
		return e.Operator + s.getFieldName(e.Expr)
	case *sqlparser.Subquery:
		return "Subquery"
	case *sqlparser.CaseExpr:
		return "CASE"
	case *sqlparser.SQLVal:
		return "Value"
	default:
		return fmt.Sprintf("Complex(%T)", expr)
	}
}

func (s *StatementMetadata) analyzeHaving(having *sqlparser.Where, sqlType string) {
	if having != nil {
		s.HasHaving = true
		s.analyzeExpr(having.Expr, sqlType)
	}
}

func (s *StatementMetadata) analyzeOrderBy(orderBy sqlparser.OrderBy, sqlType string) {
	if len(orderBy) > 0 {
		s.OrderByFields = make(map[string]int)
		for _, order := range orderBy {
			field := s.getFieldName(order.Expr)
			direction := "ASC"
			if order.Direction == sqlparser.DescScr {
				direction = "DESC"
			}
			s.OrderByFields[field+" "+direction]++

			s.analyzeExpr(order.Expr, "ORDER BY")
		}
	}
}

func (s *StatementMetadata) analyzeLimit(limit *sqlparser.Limit) {
	if limit != nil {
		s.HasLimit = true
		if limit.Offset != nil {
			s.HasOffset = true
		}
	}
}

func (s *StatementMetadata) analyzeColumns(columns sqlparser.Columns) {
	for _, col := range columns {
		s.Fields = append(s.Fields, col.String())
	}
}

func (s *StatementMetadata) analyzeUpdateExpressions(exprs sqlparser.UpdateExprs, sqlType string) {
	for _, expr := range exprs {
		s.Fields = append(s.Fields, expr.Name.Name.String())
		s.analyzeExpr(expr.Expr, sqlType)
	}
}
