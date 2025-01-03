package mybatis

import (
	"fmt"
	"log"
	"path/filepath"
	"sort"
	"strconv"
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
	rg.writeJoinAnalysis(sa)

	rg.writeBatchInsertInfo(sa)

	// TODO rg.writeBatchUpdateInfo(sa)
	// TODO rg.writeBatchDeleteInfo(sa)

	rg.writeAggregationFunctions(sa.AggregationFuncs)
	rg.writeTimeoutInfo(sa.TimeoutStatements)

	color.Cyan("Top %d most used tables", TopK)
	printTopN(sa.Tables, TopK, []string{"Table", "Count"})

	color.Cyan("Top %d most used fields", TopK)
	printTopN(sa.Fields, TopK, []string{"Field", "Count"})

	if true {
		return
	}

	rg.writeIndexRecommendations(sa)
	rg.writeSimilarityReport(sa)

	rg.writeUnknownFragments(sa.UnknownFragments)
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
}

func (rg *ReportGenerator) writeTimeoutInfo(timeoutStatements map[string]int) {
	if len(timeoutStatements) == 0 {
		return
	}

	color.Cyan("Statements with Timeout")
	header := []string{"Timeout", "Count"}
	var cellData [][]string

	for timeout, count := range timeoutStatements {
		cellData = append(cellData, []string{timeout, fmt.Sprintf("%d", count)})
	}

	sort.Slice(cellData, func(i, j int) bool {
		ti, _ := strconv.Atoi(strings.TrimSuffix(cellData[i][0], "s"))
		tj, _ := strconv.Atoi(strings.TrimSuffix(cellData[j][0], "s"))
		return ti > tj
	})

	tabular.Display(header, cellData, false, -1)
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
	if len(cellData) == 0 {
		return
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
	color.Cyan("Statements: %d", stmts)

	var items []tabular.BarChartItem
	for sqlType, count := range sqlTypes {
		items = append(items, tabular.BarChartItem{Name: sqlType, Count: count})
	}

	tabular.PrintBarChart(items, 0) // 0 means print all items
}

func (rg *ReportGenerator) writeComplexityMetrics(sa *SQLAnalyzer) {
	color.Cyan("Complexity Operation")
	header := []string{"Metric", "Count"}
	metrics := map[string]int{
		"JOIN":                 sa.JoinOperations,
		"SubQueries":           sa.SubQueries,
		"DISTINCT":             sa.DistinctQueries,
		"ORDER BY":             sa.OrderByOperations,
		"UNION":                sa.UnionOperations,
		"LIMIT with OFFSET":    sa.LimitWithOffset,
		"LIMIT without OFFSET": sa.LimitWithoutOffset,
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

// 在 report.go 中添加
func (rg *ReportGenerator) writeJoinAnalysis(sa *SQLAnalyzer) {
	// JOIN 类型统计
	joinTypeItems := make([]tabular.BarChartItem, 0, len(sa.JoinTypes))
	for joinType, count := range sa.JoinTypes {
		joinTypeItems = append(joinTypeItems, tabular.BarChartItem{Name: joinType, Count: count})
	}
	color.Cyan("Join Types")
	tabular.PrintBarChart(joinTypeItems, 0)

	// JOIN 表数量统计
	joinTableItems := make([]tabular.BarChartItem, 0, len(sa.JoinTableCounts))
	for tableCount, frequency := range sa.JoinTableCounts {
		joinTableItems = append(joinTableItems, tabular.BarChartItem{Name: fmt.Sprintf("%d tables", tableCount), Count: frequency})
	}
	color.Cyan("Join Table Counts")
	tabular.PrintBarChart(joinTableItems, 0)

	// 添加一些分析和建议
	if len(sa.JoinTableCounts) > 0 {
		maxJoinTables := 0
		for tableCount := range sa.JoinTableCounts {
			if tableCount > maxJoinTables {
				maxJoinTables = tableCount
			}
		}
		if maxJoinTables > 3 {
			color.Yellow("\nNote: There are queries joining %d tables. Consider reviewing these for potential optimization.", maxJoinTables)
		}
	}
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
