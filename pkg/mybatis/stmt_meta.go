package mybatis

type StatementMetadata struct {
	SQLType          string
	Tables           []string
	Fields           []string
	UnionOperations  int
	SubQueries       int
	AggregationFuncs map[string]int // key: function name, value: count
	HasDistinct      bool
	HasOrderBy       bool
	HasLimit         bool
	HasOffset        bool
	JoinOperations   int
	JoinTypes        map[string]int
	JoinTableCount   int
	JoinConditions   map[string]int
	IndexHints       map[string]int
	Timeout          int
	IsOptimisticLock bool
}
