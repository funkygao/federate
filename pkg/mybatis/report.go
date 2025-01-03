package mybatis

import (
	"fmt"
	"log"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"federate/pkg/util"
	"github.com/fatih/color"
)

type ReportGenerator struct{}

func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{}
}

func (rg *ReportGenerator) Generate(sa *SQLAnalyzer) {
	rg.writeSQLTypes(sa.SQLTypes)

	rg.writeComplexityMetrics(sa)

	rg.writeBatchInsertInfo(sa)

	// TODO rg.writeBatchUpdateInfo(sa)
	// TODO rg.writeBatchDeleteInfo(sa)

	rg.writeAggregationFunctions(sa.AggregationFuncs)

	color.Cyan("Join types: %d", len(sa.JoinTypes))
	printTopN(sa.JoinTypes, TopK, []string{"Joint Type", "Count"})

	color.Cyan("Top %d most used tables", TopK)
	printTopN(sa.Tables, TopK, []string{"Table", "Count"})

	color.Cyan("Top %d most used fields", TopK)
	printTopN(sa.Fields, TopK, []string{"Field", "Count"})

	rg.writeIndexRecommendations(sa)
	rg.writeSimilarityReport(sa)

	rg.writeUnknownFragments(sa.UnknownFragments)
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
}

func (rg *ReportGenerator) writeUnknownFragments(fails map[string][]SqlFragmentRef) {
	if len(fails) < 1 {
		return
	}

	header := []string{"XML", "Statement ID", "Ref SQL ID"}
	var cellData [][]string
	for path, refs := range fails {
		for _, ref := range refs {
			cellData = append(cellData, []string{filepath.Base(path), ref.StmtID, ref.Refid})
		}
	}
	color.Red("Unsupported <include refid/>: %d", len(cellData))
	tabular.Display(header, cellData, true, -1)
}

func (rg *ReportGenerator) writeUnparsableSQL(unparsableSQL []UnparsableSQL, okN int) {
	if len(unparsableSQL) == 0 {
		return
	}

	if Verbosity > 1 {
		for _, sql := range unparsableSQL {
			color.Yellow("%s %s", filepath.Base(sql.Stmt.Filename), sql.Stmt.ID)
			color.Green(sql.Stmt.Raw)
			log.Println(sql.Stmt.SQL)
			color.Red("%v", sql.Error)
		}
	}

	color.Cyan("%d Statements Fail, %d OK", len(unparsableSQL), okN)
}

func (rg *ReportGenerator) writeBatchInsertInfo(sa *SQLAnalyzer) {
	if sa.BatchInserts < 1 {
		return
	}

	color.Cyan("Total Batch Inserts %d\n", sa.BatchInserts)
	header := []string{"Column", "Count"}
	cellData := sortMapByValue(sa.BatchInsertColumns)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeSQLTypes(sqlTypes map[string]int) {
	stmts := 0
	for _, n := range sqlTypes {
		stmts += n
	}
	color.Cyan("SQL Types: %d, Statements: %d", len(sqlTypes), stmts)
	header := []string{"Type", "Count"}
	cellData := sortMapByValue(sqlTypes)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeComplexityMetrics(sa *SQLAnalyzer) {
	color.Cyan("Complexity Operation Metrics")
	header := []string{"Metric", "Count"}
	metrics := map[string]int{
		"Complex Queries": sa.ComplexQueries,
		"JOIN":            sa.JoinOperations,
		"SubQueries":      sa.SubQueries,
		"DISTINCT":        sa.DistinctQueries,
		"ORDER BY":        sa.OrderByOperations,
		"LIMIT":           sa.LimitOperations,
		"UNION":           sa.UnionOperations,
	}
	cellData := sortMapByValue(metrics)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeAggregationFunctions(aggFuncs map[string]int) {
	color.Cyan("Aggregation Functions: %d", len(aggFuncs))
	header := []string{"Function", "Count"}
	cellData := sortMapByValue(aggFuncs)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeDynamicSQLElements(elements map[string]int) {
	color.Cyan("Dynamic SQL Elements: %d", len(elements))
	header := []string{"Element", "Count"}
	cellData := sortMapByValue(elements)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeIndexRecommendations(sa *SQLAnalyzer) {
	color.Cyan("Index Recommendations per Table")
	if len(sa.IgnoredFields) > 0 {
		ignoredFields := make([]string, 0, len(sa.IgnoredFields))
		for field := range sa.IgnoredFields {
			ignoredFields = append(ignoredFields, field)
		}
		log.Printf("Note: The following fields are ignored in index recommendations: %s", strings.Join(ignoredFields, ", "))
	}

	header := []string{"Table", "N", "Index Column Comination"}
	var cellData [][]string

	for _, rec := range sa.IndexRecommendations {
		combinations := make([]struct {
			Fields string
			Count  int
		}, 0, len(rec.FieldCombinations))

		for fields, count := range rec.FieldCombinations {
			combinations = append(combinations, struct {
				Fields string
				Count  int
			}{fields, count})
		}

		sort.Slice(combinations, func(i, j int) bool {
			return combinations[i].Count > combinations[j].Count
		})

		for _, comb := range combinations[:min(TopK, len(combinations))] {
			if comb.Count < 2 {
				continue
			}

			fields := util.Truncate(comb.Fields, util.TerminalWidth()-40)
			cellData = append(cellData, []string{rec.Table, fmt.Sprintf("%d", comb.Count), fields})
		}
	}

	tabular.Display(header, cellData, true, -1)
}

func (rg *ReportGenerator) writeSimilarityReport(sa *SQLAnalyzer) {
	color.Cyan("Similar SQL Statements by Type and File")

	similarities := sa.ComputeSimilarities()

	header := []string{"SQL Type", "XML", "Similarity", "Statement ID 1", "Statement ID 2"}
	var cellData [][]string

	for sqlType, files := range similarities {
		for filename, pairs := range files {
			if len(pairs) == 0 {
				continue
			}
			for _, pair := range pairs {
				simPercentage := fmt.Sprintf("%.2f%%", pair.Similarity*100)
				if pair.Similarity < SimilarityThreshold {
					continue
				}

				cellData = append(cellData, []string{sqlType,
					strings.TrimSuffix(filepath.Base(filename), "Mapper.xml"),
					simPercentage,
					pair.ID1, pair.ID2})
			}
		}
	}
	tabular.Display(header, cellData, true, -1)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sortMapByValue(m map[string]int) [][]string {
	type kv struct {
		Key   string
		Value int
	}

	var ss []kv
	for k, v := range m {
		ss = append(ss, kv{k, v})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value > ss[j].Value
	})

	var result [][]string
	for _, kv := range ss {
		result = append(result, []string{kv.Key, fmt.Sprintf("%d", kv.Value)})
	}

	return result
}

func printTopN(m map[string]int, topK int, header []string) {
	if len(m) < topK {
		topK = len(m)
	}

	cellData := sortMapByValue(m)[:topK]
	tabular.Display(header, cellData, false, -1)
}
