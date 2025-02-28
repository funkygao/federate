package mybatis

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"federate/pkg/prompt"
	"federate/pkg/tabular"
	"federate/pkg/util"
	"github.com/fatih/color"
)

type ReportGenerator struct {
	buffer    bytes.Buffer
	inlineBar bool
	topN      int
}

func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{inlineBar: true, topN: TopK * 2}
}

func (rg *ReportGenerator) Generate(sa *Aggregator) {
	if GeneratePrompt {
		prompts := map[string]string{
			"zh":   prompt.WMSMyBatisCN,
			"en":   prompt.WMSMyBatisEN,
			"mini": prompt.WMSMyBatisMini,
		}
		fmt.Fprintf(&rg.buffer, prompts[Prompt])
		fmt.Fprintf(&rg.buffer, "\n\n## MyBatis Mapper XML Analysis Report\n\n")

		rg.inlineBar = false
		log.SetOutput(io.MultiWriter(os.Stdout, &rg.buffer))
		defer func() {
			log.SetOutput(os.Stdout)
		}()
	}

	rg.writeSQLTypes(sa.SQLTypes)
	log.Println()

	rg.writeTimeoutInfo(sa.TimeoutStatements)
	log.Println()

	rg.writeGitCommitsReport(sa.GitCommits)
	rg.writeCacheConfigReport(sa.CacheConfigs)

	rg.writeSectionHeader("Top %d/%d most used tables", TopK, len(sa.Tables))
	rg.writeSectionBody(func() {
		printTopN(sa.Tables, TopK, rg.inlineBar)
	})
	log.Println()

	rg.writeTableUsageReport(sa)
	log.Println()

	rg.writeTableRelationsReport(sa.TableRelations)
	log.Println()

	rg.writeSectionHeader("Top %d/%d SQL Fragment Reuse", TopK, len(GlobalSqlFragmentUsage))
	rg.writeSectionBody(func() {
		printTopN(GlobalSqlFragmentUsage, TopK, rg.inlineBar)
	})
	log.Println()

	rg.writeUnusedSqlFragmentReport()

	if false {
		rg.writeJoinAnalysis(sa)
		log.Println()
	}

	if ShowBatchOps {
		rg.writeBatchOperations(sa)
		log.Println()
	}

	rg.writeSubStatmentReport(sa)
	log.Println()

	if ShowIndexRecommend {
		rg.writeIndexRecommendations(sa)
		log.Println()
	}

	if ShowSimilarity {
		rg.writeSimilarityReport(sa)
		log.Println()
	}

	rg.writeAggregationFunctions(sa.AggregationFuncs)
	log.Println()

	rg.writeComplexityMetrics(sa)
	log.Println()

	rg.writeComplexQueriesReport(sa.ComplexQueries)
	log.Println()

	rg.writeLocksReport(sa.OptimisticLocks, sa.PessimisticLocks)
	log.Println()

	rg.writeParameterTypeReport(sa.ParameterTypes)
	log.Println()

	rg.writeResultTypeReport(sa.ResultTypes)
	log.Println()

	rg.writeOrderByGroupByUsageReport(sa)
	log.Println()

	rg.writeGroupByFieldsReport(sa.GroupByFields)
	log.Println()

	rg.writeOrderByFieldsReport(sa.OrderByFields)
	log.Println()

	if GeneratePrompt {
		rg.writeExtraPrompt(sa)

		promptContent := rg.buffer.String()
		if err := util.ClipboardPut(promptContent); err == nil {
			log.Printf("ChatGPT Prompt 已复制到剪贴板，约 %.2fK tokens", prompt.CountTokensInK(promptContent))
		} else {
			log.Fatalf("%v", err)
		}
	}

	rg.showErrors(sa)
}

func (rg *ReportGenerator) writeSectionHeader(format string, v ...any) {
	title := fmt.Sprintf(format, v...)
	color.Magenta(title)

	if GeneratePrompt {
		fmt.Fprintf(&rg.buffer, "### %s\n\n", title)
	}
}

func (rg *ReportGenerator) writeSectionBody(content func()) {
	if GeneratePrompt {
		fmt.Fprintf(&rg.buffer, "```\n")
		content()
		fmt.Fprintf(&rg.buffer, "```\n\n")
	} else {
		content()
	}
}

func (rg *ReportGenerator) showErrors(sa *Aggregator) {
	rg.writeUnknownFragments(sa.UnknownFragments)
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
}

// 专门给大模型的输出
func (rg *ReportGenerator) writeExtraPrompt(sa *Aggregator) {
	fmt.Fprintf(&rg.buffer, "\n### Statements with name\n\n```\n")
	if PromptSQL {
		fmt.Fprintf(&rg.buffer, "Mapper XML, Statement Name, ParameterType, ResultType, SQL\n")
	} else {
		fmt.Fprintf(&rg.buffer, "Mapper XML, Statement Name, ParameterType, ResultType\n")
	}
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		if PromptSQL {
			fmt.Fprintf(&rg.buffer, "%s, %s, %s, %s, `%s`\n", filepath.Base(stmt.Filename), stmt.ID, stmt.ParameterType, stmt.ResultType, stmt.MinimalSQL())
		} else {
			fmt.Fprintf(&rg.buffer, "%s, %s, %s, %s\n", filepath.Base(stmt.Filename), stmt.ID, stmt.ParameterType, stmt.ResultType)
		}
		return nil
	})
	fmt.Fprintf(&rg.buffer, "```\n\n")
}

func (rg *ReportGenerator) writeTimeoutInfo(timeoutStatements map[string]int) {
	if len(timeoutStatements) == 0 {
		return
	}

	var cellData [][]string

	for timeout, count := range timeoutStatements {
		cellData = append(cellData, []string{timeout, fmt.Sprintf("%d", count)})
	}

	sort.Slice(cellData, func(i, j int) bool {
		ti, _ := strconv.Atoi(strings.TrimSuffix(cellData[i][0], "s"))
		tj, _ := strconv.Atoi(strings.TrimSuffix(cellData[j][0], "s"))
		return ti > tj
	})

	header := []string{"Timeout", "Count"}
	rg.writeSectionHeader("Statements with Timeout")
	rg.writeSectionBody(func() {
		tabular.Display(header, cellData, false, -1)
	})
}

func (rg *ReportGenerator) writeUnknownFragments(fails map[string][]SqlFragmentRef) {
	if len(fails) < 1 {
		return
	}

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
	header := []string{"XML", "Statement ID", "Ref SQL ID"}
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

	color.Yellow("%d Statements Fail, %d OK", len(unparsableSQL), okN)
}

func (rg *ReportGenerator) writeUnusedSqlFragmentReport() {
	var cellData [][]string
	for name, n := range GlobalSqlFragmentUsage {
		if n == 0 {
			cellData = append(cellData, []string{name})
		}
	}
	if len(cellData) > 0 {
		rg.writeSectionHeader("Unused SQL Fragment: %d", len(cellData))
		rg.writeSectionBody(func() {
			tabular.Display([]string{"SQL Fragment"}, cellData, false, 0)
		})
	}
}

func (rg *ReportGenerator) writeBatchOperations(sa *Aggregator) {
	var cellData [][]string
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		if stmt.IsBatchOperation() {
			cellData = append(cellData, []string{strings.Trim(filepath.Base(stmt.Filename), ".xml"), stmt.ID})
		}
		return nil
	})

	if len(cellData) > 0 {
		header := []string{"XML", "Statement ID"}
		rg.writeSectionHeader("Batch Operations: %d", len(cellData))
		rg.writeSectionBody(func() {
			tabular.Display(header, cellData, true, -1)
		})
	}
}

func (rg *ReportGenerator) writeSQLTypes(sqlTypes map[string]int) {
	stmts := 0
	for _, n := range sqlTypes {
		stmts += n
	}

	var items []tabular.BarChartItem
	for sqlType, count := range sqlTypes {
		items = append(items, tabular.BarChartItem{Name: sqlType, Count: count})
	}

	rg.writeSectionHeader("Statements: %d", stmts)
	rg.writeSectionBody(func() {
		tabular.PrintBarChart(items, 0, rg.inlineBar) // 0 means print all items
	})
}

func (rg *ReportGenerator) writeComplexityMetrics(sa *Aggregator) {
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
	rg.writeSectionHeader("Complex Operations")
	rg.writeSectionBody(func() {
		header := []string{"Metric", "Count"}
		tabular.Display(header, cellData, false, -1)
	})
}

func (rg *ReportGenerator) writeAggregationFunctions(aggFuncs map[string]map[string]int) {
	cellData := sortNestedMapByValueDesc(aggFuncs)
	rg.writeSectionHeader("Aggregation Functions: %d", len(cellData))
	rg.writeSectionBody(func() {
		header := []string{"Operation", "Function", "Count"}
		tabular.Display(header, cellData, true, -1)
	})
}

func (rg *ReportGenerator) writeDynamicSQLElements(elements map[string]int) {
	cellData := sortMapByValue(elements)
	rg.writeSectionHeader("Dynamic SQL Elements: %d", len(elements))
	rg.writeSectionBody(func() {
		header := []string{"Element", "Count"}
		tabular.Display(header, cellData, false, -1)
	})
}

func (rg *ReportGenerator) writeIndexRecommendations(sa *Aggregator) {
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

	rg.writeSectionHeader("Index Recommendations per Table")
	if sa.IgnoredFields.Cardinality() > 0 {
		log.Printf("Note: The following fields are ignored in index recommendations: %s", sa.IgnoredFields)
	}

	rg.writeSectionBody(func() {
		header := []string{"Table", "N", "Index Column Comination"}
		tabular.Display(header, cellData, true, 0)
	})
}

func (rg *ReportGenerator) writeJoinAnalysis(sa *Aggregator) {
	rg.writeSectionHeader("Join Types")
	rg.writeSectionBody(func() {
		printTopN(sa.JoinTypes, 0, rg.inlineBar)
	})
	log.Println()

	// JOIN 表数量统计
	joinTableItems := make([]tabular.BarChartItem, 0, len(sa.JoinTableCounts))
	for tableCount, frequency := range sa.JoinTableCounts {
		joinTableItems = append(joinTableItems, tabular.BarChartItem{Name: fmt.Sprintf("%d tables", tableCount), Count: frequency})
	}
	if len(joinTableItems) > 1 {
		rg.writeSectionHeader("Join Table Counts") // TODO 目前数据不对
		rg.writeSectionBody(func() {
			tabular.PrintBarChart(joinTableItems, 0, rg.inlineBar)
		})
		log.Println()
	}

	if len(sa.IndexHints) > 0 {
		rg.writeSectionHeader("Index Hints")
		rg.writeSectionBody(func() {
			printTopN(sa.IndexHints, 0, rg.inlineBar)
		})
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

	rg.writeSectionHeader("Similar SQL Statements by Type and File: %d", len(cellData))
	rg.writeSectionBody(func() {
		header := []string{"SQL Type", "XML", "Similarity", "Statement ID 1", "Statement ID 2"}
		tabular.Display(header, cellData, false, -1)
	})
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

	rg.writeSectionHeader("Table Usage: %d", len(data))
	rg.writeSectionBody(func() {
		header := []string{"Table", "Total Stmt", "Select", "Single Insert", "Batch Insert", "Insert On Duplicate", "Single Update", "Batch Update", "Delete"}
		tabular.DisplayWithSummary(header, data, false, util.Range(1, len(header)), -1)
	})
}

func (rg *ReportGenerator) writeTableRelationsReport(relations []TableRelation) {
	if len(relations) == 0 {
		return
	}

	var data [][]string
	for _, r := range relations {
		data = append(data, []string{
			r.Table1,
			r.Table2,
			r.JoinType,
			//r.JoinCondition,
		})
	}
	rg.writeSectionHeader("Table Relations: %d", len(relations))
	rg.writeSectionBody(func() {
		header := []string{"Table 1", "Table 2", "Join Type"}
		tabular.Display(header, data, false, 0)
	})
}

func (rg *ReportGenerator) writeComplexQueriesReport(complexQueries []CognitiveComplexity) {
	if len(complexQueries) == 0 {
		return
	}

	var data [][]string
	for _, q := range complexQueries {
		data = append(data, []string{
			strings.Trim(filepath.Base(q.Filename), ".xml"),
			q.StatementID,
			strconv.Itoa(q.Score),
			strings.Join(q.Reasons.SortedValues(), ", "),
		})
	}
	rg.writeSectionHeader("Cognitively Complex Queries (Top %d)", rg.topN)
	rg.writeSectionBody(func() {
		header := []string{"XML", "Statement ID", "Score", "Reasons"}
		tabular.Display(header, data, false, -1)
	})
}

func (rg *ReportGenerator) writeLocksReport(optimisticLocks, pessimisticLocks []*Statement) {
	var cellData [][]string
	for _, lock := range pessimisticLocks {
		cellData = append(cellData, []string{"Pessimistic", strings.Trim(filepath.Base(lock.Filename), ".xml"), lock.ID})
	}
	for _, lock := range optimisticLocks {
		cellData = append(cellData, []string{"Optimistic", strings.Trim(filepath.Base(lock.Filename), ".xml"), lock.ID})
	}
	if len(cellData) == 0 {
		return
	}

	rg.writeSectionHeader("Locking Statements: %d", len(cellData))
	rg.writeSectionBody(func() {
		header := []string{"Lock Type", "XML", "Statement ID"}
		tabular.Display(header, cellData, true, -1)
	})
}

func (rg *ReportGenerator) writeSubStatmentReport(sa *Aggregator) {
	var cellData [][]string
	sa.WalkStatements(func(tag string, stmt *Statement) error {
		if subN := stmt.SubN(); subN > 1 {
			cellData = append(cellData, []string{strings.Trim(filepath.Base(stmt.Filename), ".xml"), stmt.ID,
				fmt.Sprintf("%d", subN), stmt.ParameterType, stmt.ResultType})
		}
		return nil
	})

	if len(cellData) == 0 {
		return
	}

	rg.writeSectionHeader("Statements With Sub Statements: %d", len(cellData))
	rg.writeSectionBody(func() {
		header := []string{"XML", "Statement ID", "Sub Stmts", "ParameterType", "ResultType"}
		tabular.Display(header, cellData, true, 0)
	})
}

func (rg *ReportGenerator) writeParameterTypeReport(parameterTypes map[string]map[string]int) {
	rg.writeSectionHeader("Parameter Types by SQL Operation")
	rg.writeSectionBody(func() {
		specials := map[string]struct{}{
			"list":    struct{}{},
			"map":     struct{}{},
			"string":  struct{}{},
			"long":    struct{}{},
			"integer": struct{}{},
			"int":     struct{}{},
		}
		var data [][]string
		for tag, types := range parameterTypes {
			for paramType, count := range types {
				if paramType == "NULL" || count < 3 {
					continue
				}
				if _, special := specials[strings.ToLower(paramType)]; special {
					paramType = color.New(color.FgYellow).Sprintf("%s", paramType)
				}

				data = append(data, []string{tag, paramType, fmt.Sprintf("%d", count)})
			}
		}
		// 按 SQL 操作类型和计数降序排序
		sort.Slice(data, func(i, j int) bool {
			if data[i][0] != data[j][0] {
				return data[i][0] < data[j][0]
			}
			countI, _ := strconv.Atoi(data[i][2])
			countJ, _ := strconv.Atoi(data[j][2])
			return countI > countJ
		})

		header := []string{"SQL Operation", "Parameter Type", "Count"}
		tabular.Display(header, data, true, -1)
	})
}

func (rg *ReportGenerator) writeResultTypeReport(resultTypes map[string]map[string]int) {
	if len(resultTypes) == 0 {
		return
	}

	rg.writeSectionHeader("Result Types by SQL Operation")
	rg.writeSectionBody(func() {
		specials := map[string]struct{}{
			"long":       struct{}{},
			"integer":    struct{}{},
			"int":        struct{}{},
			"string":     struct{}{},
			"boolean":    struct{}{},
			"bigdecimal": struct{}{},
			"map":        struct{}{},
			"date":       struct{}{},
		}
		var data [][]string
		for tag, types := range resultTypes {
			for resultType, count := range types {
				if resultType == "NULL" || count < 3 {
					continue
				}

				if _, special := specials[strings.ToLower(resultType)]; special {
					resultType = color.New(color.FgYellow).Sprintf("%s", resultType)
				}
				data = append(data, []string{tag, resultType, fmt.Sprintf("%d", count)})
			}
		}
		// 按 SQL 操作类型和计数降序排序
		sort.Slice(data, func(i, j int) bool {
			if data[i][0] != data[j][0] {
				return data[i][0] < data[j][0]
			}
			countI, _ := strconv.Atoi(data[i][2])
			countJ, _ := strconv.Atoi(data[j][2])
			return countI > countJ
		})

		header := []string{"SQL Operation", "Result Type", "Count"}
		tabular.Display(header, data, true, -1)
	})
}

func (rg *ReportGenerator) writeOrderByGroupByUsageReport(sa *Aggregator) {
	rg.writeSectionHeader("ORDER BY and GROUP BY Usage")
	rg.writeSectionBody(func() {
		totalStatements := sa.SelectOK
		data := [][]string{
			{"DISTINCT", fmt.Sprintf("%d", sa.DistinctUsage), fmt.Sprintf("%.2f%%", float64(sa.DistinctUsage)/float64(totalStatements)*100)},
			{"ORDER BY", fmt.Sprintf("%d", sa.OrderByUsage), fmt.Sprintf("%.2f%%", float64(sa.OrderByUsage)/float64(totalStatements)*100)},
			{"GROUP BY", fmt.Sprintf("%d", sa.GroupByUsage), fmt.Sprintf("%.2f%%", float64(sa.GroupByUsage)/float64(totalStatements)*100)},
		}

		header := []string{"Clause in SELECT", "Usage Count", "Usage Percentage"}
		tabular.Display(header, data, false, -1)
	})
}

func (rg *ReportGenerator) writeGroupByFieldsReport(groupByFields map[string]int) {
	n := len(groupByFields)
	if n == 0 {
		return
	}

	if n > rg.topN {
		n = rg.topN
	}
	rg.writeSectionHeader("Top %d/%d GROUP BY Field Combinations Usage", n, len(groupByFields))
	rg.writeSectionBody(func() {
		var data [][]string
		for field, count := range groupByFields {
			data = append(data, []string{field, fmt.Sprintf("%d", count)})
		}

		// 按使用次数降序排序
		sort.Slice(data, func(i, j int) bool {
			countI, _ := strconv.Atoi(data[i][1])
			countJ, _ := strconv.Atoi(data[j][1])
			return countI > countJ
		})

		header := []string{"Field Combination", "Usage Count"}
		tabular.Display(header, data[:n], false, -1)
	})
}

func (rg *ReportGenerator) writeOrderByFieldsReport(orderByFields map[string]int) {
	n := len(orderByFields)
	if n > rg.topN {
		n = rg.topN
	}
	rg.writeSectionHeader("Top %d/%d ORDER BY Fields Usage", n, len(orderByFields))
	rg.writeSectionBody(func() {
		var data [][]string

		for fieldAndDirection, count := range orderByFields {
			field, direction := splitFieldAndDirection(fieldAndDirection)
			data = append(data, []string{field, direction, fmt.Sprintf("%d", count)})
		}

		// 首先按使用次数降序排序，然后按字段名字母顺序排序，最后按方向排序
		sort.Slice(data, func(i, j int) bool {
			countI, _ := strconv.Atoi(data[i][2])
			countJ, _ := strconv.Atoi(data[j][2])
			if countI != countJ {
				return countI > countJ
			}
			if data[i][0] != data[j][0] {
				return data[i][0] < data[j][0]
			}
			return data[i][1] < data[j][1]
		})

		header := []string{"Field", "Direction", "Usage Count"}
		tabular.Display(header, data[:n], false, -1)
	})
}

func splitFieldAndDirection(fieldAndDirection string) (string, string) {
	lastSpaceIndex := strings.LastIndex(fieldAndDirection, " ")
	if lastSpaceIndex == -1 {
		return fieldAndDirection, "ASC"
	}

	field := fieldAndDirection[:lastSpaceIndex]
	direction := fieldAndDirection[lastSpaceIndex+1:]

	// 处理复杂表达式
	if strings.HasPrefix(field, "(") && !strings.HasSuffix(field, ")") {
		field = fieldAndDirection[:strings.LastIndex(fieldAndDirection, ")")+1]
		direction = strings.TrimSpace(fieldAndDirection[strings.LastIndex(fieldAndDirection, ")")+1:])
	}

	// 处理 JSON 操作符
	if strings.Contains(field, "->") {
		parts := strings.SplitN(field, "->", 2)
		field = parts[0]
		if len(parts) > 1 {
			direction = "-> " + strings.TrimSpace(parts[1]) + " " + direction
		} else {
			direction = "-> " + direction
		}
	}

	return strings.TrimSpace(field), strings.TrimSpace(direction)
}

func (rg *ReportGenerator) writeCacheConfigReport(cacheConfigs map[string]*CacheConfig) {
	if len(cacheConfigs) == 0 {
		return
	}

	rg.writeSectionHeader("Cache Configurations: %d", len(cacheConfigs))
	rg.writeSectionBody(func() {
		var data [][]string
		for namespace, config := range cacheConfigs {
			data = append(data, []string{
				namespace,
				config.Type,
				config.EvictionPolicy,
				config.FlushInterval,
				config.Size,
				strconv.FormatBool(config.ReadOnly),
				strconv.FormatBool(config.Blocking),
			})
		}
		// 按命名空间排序
		sort.Slice(data, func(i, j int) bool {
			return data[i][0] < data[j][0]
		})

		header := []string{"Namespace", "Type", "Eviction Policy", "Flush Interval", "Size", "Read Only", "Blocking"}
		tabular.Display(header, data, false, -1)
	})
	log.Println()
}

func (rg *ReportGenerator) writeGitCommitsReport(gitCommits map[string]int) {
	if len(gitCommits) == 0 {
		return
	}

	n := len(gitCommits)
	if n > rg.topN {
		n = rg.topN
	}
	rg.writeSectionHeader("Top %d/%d Git Commit Statistics", n, len(gitCommits))
	rg.writeSectionBody(func() {
		var data [][]string
		for file, count := range gitCommits {
			data = append(data, []string{strings.Trim(filepath.Base(file), ".xml"), strconv.Itoa(count)})
		}

		// 按提交次数降序排序
		sort.Slice(data, func(i, j int) bool {
			countI, _ := strconv.Atoi(data[i][1])
			countJ, _ := strconv.Atoi(data[j][1])
			return countI > countJ
		})

		header := []string{"XML", "Commit Count"}
		tabular.Display(header, data[:n], false, -1)
	})
	log.Println()
}
