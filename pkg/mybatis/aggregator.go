package mybatis

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"federate/pkg/primitive"
)

type Aggregator struct {
	StatementsByTag map[string][]*Statement

	// config
	IgnoredFields *primitive.StringSet

	// metrics
	SQLTypes             map[string]int
	Tables               map[string]int
	Fields               map[string]int
	UnionOperations      int
	SubQueries           int
	AggregationFuncs     map[string]map[string]int
	DistinctQueries      int
	OrderByOperations    int
	LimitOperations      int
	LimitWithOffset      int
	LimitWithoutOffset   int
	JoinOperations       int
	JoinTypes            map[string]int
	JoinTableCounts      map[int]int
	JoinConditions       map[string]int
	IndexHints           map[string]int
	ParsedOK             int
	TimeoutStatements    map[string]int
	IndexRecommendations map[string]*TableIndexRecommendation
	TableUsage           map[string]*TableUsage
	TableRelations       []TableRelation
	ComplexQueries       []CognitiveComplexity
	OptimisticLocks      []*Statement
	ParameterTypes       map[string]map[string]int // Tag -> ParameterType -> Count
	ResultTypes          map[string]map[string]int // Tag -> ResultType -> Count

	// errors
	UnknownFragments map[string][]SqlFragmentRef
	UnparsableSQL    []UnparsableSQL
}

func NewAggregator(ignoredFields []string) *Aggregator {
	return &Aggregator{
		IgnoredFields:        primitive.NewStringSet().Add(ignoredFields...),
		SQLTypes:             make(map[string]int),
		Tables:               make(map[string]int),
		Fields:               make(map[string]int),
		AggregationFuncs:     make(map[string]map[string]int),
		JoinTypes:            make(map[string]int),
		JoinTableCounts:      make(map[int]int),
		JoinConditions:       make(map[string]int),
		ParameterTypes:       make(map[string]map[string]int),
		ResultTypes:          make(map[string]map[string]int),
		IndexHints:           make(map[string]int),
		TimeoutStatements:    make(map[string]int),
		StatementsByTag:      make(map[string][]*Statement),
		IndexRecommendations: make(map[string]*TableIndexRecommendation),
		UnparsableSQL:        []UnparsableSQL{},
		UnknownFragments:     make(map[string][]SqlFragmentRef),
	}
}

func (sa *Aggregator) Aggregate() {
	sa.analyzeTableUsage()
	sa.analyzeTableRelations()
	sa.analyzeCognitiveComplexity()
	sa.detectOptimisticLocking()
}

func (sa *Aggregator) OnStmt(s Statement) error {
	if s.Timeout > 0 {
		timeoutKey := fmt.Sprintf("%ds", s.Timeout)
		sa.TimeoutStatements[timeoutKey]++
	}

	sa.addStatement(&s)

	if err := s.Analyze(); err != nil {
		sa.UnparsableSQL = append(sa.UnparsableSQL, UnparsableSQL{s, err})
		return err
	}

	sa.ParsedOK++
	sa.updateAggregatedMetrics(&s)

	if sa.ParameterTypes[s.Tag] == nil {
		sa.ParameterTypes[s.Tag] = make(map[string]int)
	}
	sa.ParameterTypes[s.Tag][s.ParameterType]++

	if sa.ResultTypes[s.Tag] == nil {
		sa.ResultTypes[s.Tag] = make(map[string]int)
	}
	sa.ResultTypes[s.Tag][s.ResultType]++

	return nil
}

// WalkStatements 遍历 Aggregator 中的所有语句
func (sa *Aggregator) WalkStatements(walkFn func(tag string, stmt *Statement) error) error {
	for tag, stmts := range sa.StatementsByTag {
		for _, stmt := range stmts {
			if err := walkFn(tag, stmt); err != nil {
				return err
			}
		}
	}
	return nil
}

func (sa *Aggregator) addStatement(s *Statement) {
	sa.StatementsByTag[s.Tag] = append(sa.StatementsByTag[s.Tag], s)
}

func (sa *Aggregator) updateAggregatedMetrics(stmt *Statement) {
	sa.ComplexQueries = append(sa.ComplexQueries, stmt.Complexity)

	s := stmt.Metadata

	for _, sqlType := range s.SQLTypes {
		sa.SQLTypes[sqlType]++
	}

	for _, table := range s.Tables {
		sa.Tables[table]++
	}

	for _, field := range s.Fields {
		sa.Fields[field]++
	}

	sa.UnionOperations += s.UnionOperations
	sa.SubQueries += s.SubQueries

	for sqlType, funcMap := range s.AggregationFuncs {
		if sa.AggregationFuncs[sqlType] == nil {
			sa.AggregationFuncs[sqlType] = make(map[string]int)
		}
		for funcName, count := range funcMap {
			sa.AggregationFuncs[sqlType][funcName] += count
		}
	}

	for table, rec := range stmt.Metadata.IndexRecommendations {
		if sa.IndexRecommendations[table] == nil {
			sa.IndexRecommendations[table] = &TableIndexRecommendation{
				Table:             table,
				FieldCombinations: make(map[string]int),
				JoinFields:        make(map[string]int),
				WhereFields:       make(map[string]int),
				GroupByFields:     make(map[string]int),
				OrderByFields:     make(map[string]int),
			}
		}

		saRec := sa.IndexRecommendations[table]
		for field, count := range rec.JoinFields {
			saRec.JoinFields[field] += count
		}
		for field, count := range rec.WhereFields {
			saRec.WhereFields[field] += count
		}
		for field, count := range rec.GroupByFields {
			saRec.GroupByFields[field] += count
		}
		for field, count := range rec.OrderByFields {
			saRec.OrderByFields[field] += count
		}
		for combination, count := range rec.FieldCombinations {
			saRec.FieldCombinations[combination] += count
		}
	}

	if s.HasDistinct {
		sa.DistinctQueries++
	}

	if s.HasOrderBy {
		sa.OrderByOperations++
	}

	if s.HasLimit {
		sa.LimitOperations++
		if s.HasOffset {
			sa.LimitWithOffset++
		} else {
			sa.LimitWithoutOffset++
		}
	}

	sa.JoinOperations += s.JoinOperations

	for joinType, count := range s.JoinTypes {
		sa.JoinTypes[joinType] += count
	}

	sa.JoinTableCounts[s.JoinTableCount]++

	for condition, count := range s.JoinConditions {
		sa.JoinConditions[condition] += count
	}

	for hint, count := range s.IndexHints {
		sa.IndexHints[hint] += count
	}

	if stmt.HasOptimisticLocking() {
		sa.OptimisticLocks = append(sa.OptimisticLocks, stmt)
	}
}

// TODO 逻辑不大对，表可能被重复计算
func (sa *Aggregator) analyzeTableUsage() {
	sa.TableUsage = make(map[string]*TableUsage)
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		for _, table := range stmt.Metadata.Tables {
			if table == "SUBQUERY" {
				continue // 跳过子查询
			}

			if _, ok := sa.TableUsage[table]; !ok {
				sa.TableUsage[table] = &TableUsage{Name: table}
			}
			sa.TableUsage[table].UseCount++
			switch tag {
			case "select":
				sa.TableUsage[table].InSelect++
			case "insert":
				if strings.Contains(stmt.SQL, table) {
					if stmt.IsBatchOperation() {
						sa.TableUsage[table].BatchInsert++
					} else {
						sa.TableUsage[table].SingleInsert++
					}
					if stmt.HasOnDuplicateKey() {
						sa.TableUsage[table].InsertOnDuplicate++
					}
				}
			case "update":
				if strings.Contains(stmt.SQL, table) {
					if stmt.IsBatchOperation() {
						sa.TableUsage[table].BatchUpdate++
					} else {
						sa.TableUsage[table].SingleUpdate++
					}
				}
			case "delete":
				sa.TableUsage[table].InDelete++
			}
		}
		return nil
	})
}

func (sa *Aggregator) analyzeTableRelations() {
	relationMap := make(map[string]TableRelation)
	sa.WalkStatements(func(tag string, stmt *Statement) error {
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
		return nil
	})

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

func (sa *Aggregator) analyzeCognitiveComplexity() {
	// 按复杂度得分降序排序
	sort.Slice(sa.ComplexQueries, func(i, j int) bool {
		return sa.ComplexQueries[i].Score > sa.ComplexQueries[j].Score
	})

	// 只保留前 TopK 个复杂查询
	if len(sa.ComplexQueries) > TopK {
		sa.ComplexQueries = sa.ComplexQueries[:TopK]
	}
}

func (sa *Aggregator) detectOptimisticLocking() {
	sa.OptimisticLocks = []*Statement{}
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		if stmt.HasOptimisticLocking() {
			sa.OptimisticLocks = append(sa.OptimisticLocks, stmt)
		}
		return nil
	})
}

type SimilarityPair struct {
	ID1        string
	ID2        string
	Similarity float64
}

func (sa *Aggregator) ComputeSimilarities() map[string]map[string][]SimilarityPair {
	result := make(map[string]map[string][]SimilarityPair) // map[SQLType]map[Filename][]SimilarityPair

	for sqlType, statements := range sa.StatementsByTag {
		// 初始化内层映射
		if _, ok := result[sqlType]; !ok {
			result[sqlType] = make(map[string][]SimilarityPair)
		}

		// 按文件名分组语句
		statementsByFile := make(map[string][]*Statement)
		for _, stmt := range statements {
			filename := stmt.Filename
			statementsByFile[filename] = append(statementsByFile[filename], stmt)
		}

		// 在每个文件中计算相似度
		for filename, stmtsInFile := range statementsByFile {
			var pairs []SimilarityPair
			// 对文件中的所有语句进行标记化和规范化
			tokensList := make([][]string, len(stmtsInFile))
			for i, stmt := range stmtsInFile {
				tokensList[i] = tokenizeAndNormalize(stmt.SQL)
			}

			// 计算两两之间的相似度
			for i := 0; i < len(stmtsInFile); i++ {
				for j := i + 1; j < len(stmtsInFile); j++ {
					similarity := computeJaccardSimilarity(tokensList[i], tokensList[j])
					pair := SimilarityPair{
						ID1:        stmtsInFile[i].ID,
						ID2:        stmtsInFile[j].ID,
						Similarity: similarity,
					}
					pairs = append(pairs, pair)
				}
			}

			// 按相似度排序
			sort.Slice(pairs, func(i, j int) bool {
				return pairs[i].Similarity > pairs[j].Similarity
			})

			// 每个XML文件内，选择前 N 个相似度最高的
			if len(pairs) > TopK {
				pairs = pairs[:TopK]
			}

			// 将结果存入返回值
			result[sqlType][filename] = pairs
		}
	}

	return result
}

func tokenizeAndNormalize(sql string) []string {
	// Remove literals and placeholders
	sql = regexp.MustCompile(`'[^']*'|\?`).ReplaceAllString(sql, "")
	// Convert to lowercase
	sql = strings.ToLower(sql)
	// Split into tokens (simple split by whitespace and punctuation)
	tokens := regexp.MustCompile(`\W+`).Split(sql, -1)
	// Remove empty tokens
	var normalizedTokens []string
	for _, token := range tokens {
		if token != "" {
			normalizedTokens = append(normalizedTokens, token)
		}
	}
	return normalizedTokens
}

func computeJaccardSimilarity(tokens1, tokens2 []string) float64 {
	set1 := make(map[string]struct{})
	set2 := make(map[string]struct{})
	for _, token := range tokens1 {
		set1[token] = struct{}{}
	}
	for _, token := range tokens2 {
		set2[token] = struct{}{}
	}
	var intersectionSize int
	for token := range set1 {
		if _, exists := set2[token]; exists {
			intersectionSize++
		}
	}
	unionSize := len(set1) + len(set2) - intersectionSize
	if unionSize == 0 {
		return 0
	}
	return float64(intersectionSize) / float64(unionSize)
}
