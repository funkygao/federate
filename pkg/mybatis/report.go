package mybatis

import (
	"fmt"
	"log"
	"path/filepath"
	"sort"

	"federate/pkg/tabular"
)

type ReportGenerator struct{}

func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{}
}

func (rg *ReportGenerator) Generate(sa *SQLAnalyzer) {
	rg.writeUnparsableSQL(sa.UnparsableSQL, sa.ParsedOK)
	rg.writeSQLTypes(sa.SQLTypes)
	//rg.writeMostUsedTables(sa.Tables)
	//rg.writeMostUsedFields(sa.Fields)
	rg.writeComplexityMetrics(sa)
	rg.writeAggregationFunctions(sa.AggregationFuncs)

	topK := 20

	log.Printf("Join types:")
	printTopN(sa.JoinTypes, topK, []string{"Joint Type", "Count"})

	log.Printf("Top %d most used tables:", topK)
	printTopN(sa.Tables, topK, []string{"Table", "Count"})

	log.Printf("Top %d most used fields:", topK)
	printTopN(sa.Fields, topK, []string{"Field", "Count"})

	log.Printf("Top %d index recommendations:", topK)
	printTopN(sa.IndexRecommendations, topK, []string{"Field", "Count"})
}

func (rg *ReportGenerator) writeUnparsableSQL(unparsableSQL []UnparsableSQL, okN int) {
	if len(unparsableSQL) == 0 {
		return
	}

	log.Println("Unparsable SQL Statements:")
	for _, sql := range unparsableSQL {
		log.Printf("%s %s\n%s\n%v", filepath.Base(sql.Stmt.Filename), sql.Stmt.ID, sql.Stmt.ParseableSQL, sql.Error)
	}

	log.Printf("%d Statements Fail, %d OK", len(unparsableSQL), okN)
	log.Println()
}

func (rg *ReportGenerator) writeBatchInsertInfo(sa *SQLAnalyzer) {
	log.Println("Batch Insert Operations:")
	log.Printf("Total Batch Inserts: %d\n", sa.BatchInserts)
	log.Println("Columns used in Batch Inserts:")
	header := []string{"Column", "Count"}
	cellData := sortMapByValue(sa.BatchInsertColumns)
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeSQLTypes(sqlTypes map[string]int) {
	log.Println("SQL Types:")
	header := []string{"Type", "Count"}
	cellData := sortMapByValue(sqlTypes)
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeMostUsedTables(tables map[string]int) {
	log.Println("Most Used Tables:")
	header := []string{"Table", "Count"}
	cellData := sortMapByValue(tables)
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeMostUsedFields(fields map[string]int) {
	log.Println("Most Used Fields:")
	header := []string{"Field", "Count"}
	cellData := sortMapByValue(fields)
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeComplexityMetrics(sa *SQLAnalyzer) {
	log.Println("Complexity Operation Metrics:")
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
	log.Println()
}

func (rg *ReportGenerator) writeAggregationFunctions(aggFuncs map[string]int) {
	log.Println("Aggregation Functions:")
	header := []string{"Function", "Count"}
	cellData := sortMapByValue(aggFuncs)
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeDynamicSQLElements(elements map[string]int) {
	log.Println("Dynamic SQL Elements:")
	header := []string{"Element", "Count"}
	cellData := sortMapByValue(elements)
	tabular.Display(header, cellData, false, -1)
	log.Println()
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
	log.Println()
}
