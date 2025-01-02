package mybatis

import (
	"fmt"
	"log"
	"path/filepath"
	"sort"

	"federate/pkg/tabular"
	"github.com/fatih/color"
)

type ReportGenerator struct {
	verbosity int
}

func NewReportGenerator(verbosity int) *ReportGenerator {
	return &ReportGenerator{verbosity}
}

func (rg *ReportGenerator) Generate(sa *SQLAnalyzer, topK int) {
	rg.writeSQLTypes(sa.SQLTypes)

	rg.writeComplexityMetrics(sa)

	rg.writeBatchInsertInfo(sa)

	rg.writeAggregationFunctions(sa.AggregationFuncs)

	color.Cyan("Join types")
	printTopN(sa.JoinTypes, topK, []string{"Joint Type", "Count"})

	color.Cyan("Top %d most used tables", topK)
	printTopN(sa.Tables, topK, []string{"Table", "Count"})

	color.Cyan("Top %d most used fields", topK)
	printTopN(sa.Fields, topK, []string{"Field", "Count"})

	color.Cyan("Top %d index recommendations", topK)
	printTopN(sa.IndexRecommendations, topK, []string{"Field", "Count"})

	rg.writeUnknownFragments(sa.UnknownFragments)
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
}

func (rg *ReportGenerator) writeUnknownFragments(fails map[string][]SqlFragmentRef) {
	if len(fails) < 1 {
		return
	}

	color.Red("Unsupported <include refid/>")
	header := []string{"XML", "Statement ID", "Ref SQL ID"}
	var cellData [][]string
	for path, refs := range fails {
		for _, ref := range refs {
			cellData = append(cellData, []string{filepath.Base(path), ref.StmtID, ref.Refid})
		}
	}
	tabular.Display(header, cellData, true, -1)
}

func (rg *ReportGenerator) writeUnparsableSQL(unparsableSQL []UnparsableSQL, okN int) {
	if len(unparsableSQL) == 0 {
		return
	}

	if rg.verbosity > 1 {
		color.Red("Unparsable SQL Statements")
		for _, sql := range unparsableSQL {
			log.Printf("%s %s\n%s\n%v", filepath.Base(sql.Stmt.Filename), sql.Stmt.ID, sql.Stmt.ParseableSQL, sql.Error)
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
	color.Cyan("SQL Types")
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
	color.Cyan("Aggregation Functions")
	header := []string{"Function", "Count"}
	cellData := sortMapByValue(aggFuncs)
	tabular.Display(header, cellData, false, -1)
}

func (rg *ReportGenerator) writeDynamicSQLElements(elements map[string]int) {
	color.Cyan("Dynamic SQL Elements")
	header := []string{"Element", "Count"}
	cellData := sortMapByValue(elements)
	tabular.Display(header, cellData, false, -1)
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
