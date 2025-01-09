package mybatis

import (
	"log"

	"github.com/xwb1989/sqlparser"
)

type Visitor[T sqlparser.SQLNode] interface {

	// Visit is called for each node of type T in the AST.
	Visit(T) (continueTraversal bool)
}

func walkAST[T sqlparser.SQLNode](visitor Visitor[T], node T) {
	sqlparser.Walk(func(n sqlparser.SQLNode) (kontinue bool, err error) {
		return visitor.Visit(n.(T)), nil
	}, node)
}

type SQLTypeVisitor struct {
	Metadata StatementMetadata
}

func (s *Statement) NewSQLTypeVisitor() *SQLTypeVisitor {
	return &SQLTypeVisitor{s.Metadata}
}

func (v *SQLTypeVisitor) Visit(node sqlparser.SQLNode) bool {
	switch stmt := node.(type) {
	case *sqlparser.Select:
		v.Metadata.addSQLType("SELECT")
	case *sqlparser.Insert:
		v.Metadata.addSQLType("INSERT")
	case *sqlparser.Update:
		v.Metadata.addSQLType("UPDATE")
	case *sqlparser.Delete:
		v.Metadata.addSQLType("DELETE")
	case *sqlparser.Union:
		v.Metadata.addSQLType("UNION")
	case *sqlparser.Set:
		v.Metadata.addSQLType("Set @")
	default:
		log.Printf("unknown stmt type: %T", stmt)
	}

	return true
}
