package mybatis

import (
	"log"
	"strings"

	"github.com/xwb1989/sqlparser"
)

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

func containsWildcard(expr sqlparser.Expr) bool {
	switch e := expr.(type) {
	case *sqlparser.SQLVal:
		if e.Type == sqlparser.StrVal {
			return strings.Contains(string(e.Val), "%")
		}
	case *sqlparser.ColName:
		// 列名不包含通配符
	case *sqlparser.FuncExpr:
		// 函数表达式不包含通配符
	default:
		log.Printf("Unhandled expr type for wildcard check: %T", e)
	}
	return false
}
