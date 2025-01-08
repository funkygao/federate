package mybatis

import (
	"log"
	"strings"

	"github.com/xwb1989/sqlparser"
)

// recursive
func (s *Statement) analyzeExpr(expr sqlparser.Expr, complexity *CognitiveComplexity) {
	switch e := expr.(type) {
	case *sqlparser.AndExpr:
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.OrExpr:
		complexity.Reasons.Add("OR")
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.ComparisonExpr:
		if e.Operator == sqlparser.LikeStr {
		}
		s.analyzeExpr(e.Left, complexity)
		s.analyzeExpr(e.Right, complexity)
	case *sqlparser.Subquery:
		complexity.Reasons.Add("Subquery")
		s.analyzeNode(e.Select, complexity)
	case *sqlparser.FuncExpr:
		s.analyzeFunction(e, complexity)
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
	case sqlparser.ListArg, *sqlparser.SQLVal, *sqlparser.BinaryExpr, *sqlparser.UnaryExpr,
		*sqlparser.ColName,
		*sqlparser.GroupConcatExpr, sqlparser.BoolVal, *sqlparser.NullVal, *sqlparser.ConvertExpr:
		// 简单，不增加复杂度
	default:
		log.Printf("Unhandled expr type: %T", e)
	}
}

func (s *Statement) analyzeCase(caseExpr *sqlparser.CaseExpr, complexity *CognitiveComplexity) {
	complexity.Reasons.Add("CASE")

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

func (s *Statement) analyzeFunction(funcExpr *sqlparser.FuncExpr, complexity *CognitiveComplexity) {
	switch strings.ToUpper(funcExpr.Name.String()) {
	case "MIN", "MAX", "AVG", "SUM", "COUNT":
		complexity.Reasons.Add(strings.ToUpper(funcExpr.Name.String()))
	}
}
