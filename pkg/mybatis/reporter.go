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

func (rg *ReportGenerator) Generate(sa *Aggregator) {
	rg.writeSQLTypes(sa.SQLTypes)
	log.Println()

	color.Magenta("Top %d most used tables", TopK)
	printTopN(sa.Tables, TopK)
	log.Println()

	rg.writeJoinAnalysis(sa)
	log.Println()

	rg.writeAggregationFunctions(sa.AggregationFuncs)
	log.Println()

	rg.writeTimeoutInfo(sa.TimeoutStatements)
	log.Println()

	if ShowBatchOps {
		rg.writeBatchOperations(sa)
		log.Println()
	}

	if ShowIndexRecommend {
		rg.writeIndexRecommendations(sa)
		log.Println()
	}

	if ShowSimilarity {
		rg.writeSimilarityReport(sa)
		log.Println()
	}

	rg.writeTableUsageReport(sa)
	log.Println()

	rg.writeTableRelationsReport(sa.TableRelations)
	log.Println()

	rg.writeComplexityMetrics(sa)
	log.Println()

	rg.writeComplexQueriesReport(sa.ComplexQueries)
	log.Println()

	rg.writeOptimisticLocksReport(sa.OptimisticLocks)
	log.Println()

	rg.writeSubStatmentReport(sa)
	log.Println()

	rg.showErrors(sa)
}

func (rg *ReportGenerator) showErrors(sa *Aggregator) {
	rg.writeUnknownFragments(sa.UnknownFragments)
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
}

func (rg *ReportGenerator) writeTimeoutInfo(timeoutStatements map[string]int) {
	if len(timeoutStatements) == 0 {
		return
	}

	color.Magenta("Statements with Timeout")
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
			color.Green(sql.Stmt.XMLText)
			log.Println(sql.Stmt.SQL)
			color.Red("%v", sql.Error)
		}
	}

	color.Magenta("%d Statements Fail, %d OK", len(unparsableSQL), okN)
}

func (rg *ReportGenerator) writeBatchOperations(sa *Aggregator) {
	header := []string{"XML", "Statement ID"}
	var cellData [][]string

	sa.WalkStatements(func(tag string, stmt *Statement) error {
		if stmt.IsBatchOperation() {
			cellData = append(cellData, []string{strings.Trim(filepath.Base(stmt.Filename), ".xml"), stmt.ID})
		}
		return nil
	})

	if len(cellData) > 0 {
		color.Magenta("Batch Operations: %d", len(cellData))
		tabular.Display(header, cellData, true, -1)
	}
}

func (rg *ReportGenerator) writeSQLTypes(sqlTypes map[string]int) {
	stmts := 0
	for _, n := range sqlTypes {
		stmts += n
	}
	color.Magenta("Statements: %d", stmts)

	var items []tabular.BarChartItem
	for sqlType, count := range sqlTypes {
		items = append(items, tabular.BarChartItem{Name: sqlType, Count: count})
	}

	tabular.PrintBarChart(items, 0) // 0 means print all items
}

func (rg *ReportGenerator) writeComplexityMetrics(sa *Aggregator) {
	color.Magenta("Complexity Operation")
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

func (rg *ReportGenerator) writeAggregationFunctions(aggFuncs map[string]map[string]int) {
	header := []string{"Operation", "Function", "Count"}
	cellData := sortNestedMapByValueDesc(aggFuncs)
	color.Magenta("Aggregation Functions: %d", len(cellData))
	tabular.Display(header, cellData, true, -1)
}

func (rg *ReportGenerator) writeDynamicSQLElements(elements map[string]int) {
	color.Magenta("Dynamic SQL Elements: %d", len(elements))
	header := []string{"Element", "Count"}
	cellData := sortMapByValue(elements)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeIndexRecommendations(sa *Aggregator) {
	color.Magenta("Index Recommendations per Table")
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

	tabular.Display(header, cellData, true, 0)
}

// 在 report.go 中添加
func (rg *ReportGenerator) writeJoinAnalysis(sa *Aggregator) {
	color.Magenta("Join Types")
	printTopN(sa.JoinTypes, 0)
	log.Println()

	color.Magenta("Join Table Counts")
	// JOIN 表数量统计
	joinTableItems := make([]tabular.BarChartItem, 0, len(sa.JoinTableCounts))
	for tableCount, frequency := range sa.JoinTableCounts {
		joinTableItems = append(joinTableItems, tabular.BarChartItem{Name: fmt.Sprintf("%d tables", tableCount), Count: frequency})
	}
	tabular.PrintBarChart(joinTableItems, 0)
	log.Println()

	if len(sa.IndexHints) > 0 {
		color.Magenta("Index Hints")
		printTopN(sa.IndexHints, 0)
	}
}

func (rg *ReportGenerator) writeSimilarityReport(sa *Aggregator) {
	similarities := sa.ComputeSimilarities()
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
	color.Magenta("Similar SQL Statements by Type and File: %d", len(cellData))
	header := []string{"SQL Type", "XML", "Similarity", "Statement ID 1", "Statement ID 2"}

	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeTableUsageReport(sa *Aggregator) {
	var items []TableUsage
	for _, u := range sa.TableUsage {
		items = append(items, *u)
	}

	// 按总使用次数降序排序
	sort.Slice(items, func(i, j int) bool {
		return items[i].UseCount > items[j].UseCount
	})

	var data [][]string
	for _, item := range items {
		insertOnDuplicateStr := strconv.Itoa(item.InsertOnDuplicate)
		if item.InsertOnDuplicate > 0 {
			insertOnDuplicateStr = color.New(color.FgYellow).Sprintf("%d", item.InsertOnDuplicate)
		}
		batchInsertStr := strconv.Itoa(item.BatchInsert)
		if item.BatchInsert > 0 {
			batchInsertStr = color.New(color.FgYellow).Sprintf("%d", item.BatchInsert)
		}
		batchUpdateStr := strconv.Itoa(item.BatchUpdate)
		if item.BatchUpdate > 0 {
			batchUpdateStr = color.New(color.FgYellow).Sprintf("%d", item.BatchUpdate)
		}

		data = append(data, []string{
			item.Name,
			strconv.Itoa(item.UseCount),
			strconv.Itoa(item.InSelect),
			strconv.Itoa(item.SingleInsert),
			batchInsertStr,
			insertOnDuplicateStr,
			strconv.Itoa(item.SingleUpdate),
			batchUpdateStr,
			strconv.Itoa(item.InDelete),
		})
	}

	color.Magenta("Table Usage Analysis: %d", len(data))
	header := []string{"Table", "Total Stmt", "Select", "Single Insert", "Batch Insert", "Insert On Duplicate", "Single Update", "Batch Update", "Delete"}
	tabular.DisplayWithSummary(header, data, false, util.Range(1, len(header)), -1)
}

func (rg *ReportGenerator) writeTableRelationsReport(relations []TableRelation) {
	color.Magenta("Table Relations Analysis: %d", len(relations))
	header := []string{"Table 1", "Table 2", "Join Type"}
	var data [][]string
	for _, r := range relations {
		data = append(data, []string{
			r.Table1,
			r.Table2,
			r.JoinType,
			//r.JoinCondition,
		})
	}
	tabular.Display(header, data, false, 0)
}

func (rg *ReportGenerator) writeComplexQueriesReport(complexQueries []SQLComplexity) {
	if len(complexQueries) == 0 {
		return
	}

	color.Magenta("Complex Queries Analysis (Top %d)", TopK)
	header := []string{"XML", "Statement ID", "Score", "Reasons"}
	var data [][]string
	for _, q := range complexQueries {
		data = append(data, []string{
			strings.Trim(filepath.Base(q.Filename), ".xml"),
			q.StatementID,
			strconv.Itoa(q.Score),
			strings.Join(q.Reasons.SortedValues(), ", "),
		})
	}
	tabular.Display(header, data, false, -1)
}

func (rg *ReportGenerator) writeOptimisticLocksReport(locks []*Statement) {
	var cellData [][]string
	for _, lock := range locks {
		cellData = append(cellData, []string{
			strings.Trim(filepath.Base(lock.Filename), ".xml"),
			lock.ID,
		})
	}
	color.Magenta("Optimistic Locking: %d", len(cellData))
	header := []string{"XML", "Statement ID"}
	tabular.Display(header, cellData, true, -1)
}

func (rg *ReportGenerator) writeSubStatmentReport(sa *Aggregator) {
	var cellData [][]string
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		subN := stmt.SubN()
		if subN > 1 {
			cellData = append(cellData, []string{strings.Trim(filepath.Base(stmt.Filename), ".xml"), stmt.ID, fmt.Sprintf("%d", subN)})
		}
		return nil
	})
	color.Magenta("Statements With Sub Statements: %d", len(cellData))
	header := []string{"XML", "Statement ID", "Sub Stmts"}
	tabular.Display(header, cellData, true, 0)
}
