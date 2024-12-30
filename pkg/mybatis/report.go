package mybatis

import (
	"fmt"
	"log"
	"path/filepath"
	"sort"

	"federate/pkg/primitive"
	"federate/pkg/tabular"
)

type ReportGenerator struct{}

func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{}
}

func (rg *ReportGenerator) Generate(xa *XMLAnalyzer, sa *SQLAnalyzer) {
	//rg.writeUnparsableSQL(sa.UnparsableSQL)
	rg.writeIgnoredTags(sa.IgnoredTags)
	rg.writeSQLTypes(sa.SQLTypes)
	//rg.writeMostUsedTables(sa.Tables)
	//rg.writeMostUsedFields(sa.Fields)
	rg.writeComplexityMetrics(sa)
	rg.writeAggregationFunctions(sa.AggregationFuncs)
	rg.writeDynamicSQLElements(xa.DynamicSQLElements)

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

func (rg *ReportGenerator) writeIgnoredTags(ignored *primitive.StringSet) {
	log.Println("Ignored Tags:")
	header := []string{"Tag"}
	var cellData [][]string
	for _, tag := range ignored.Values() {
		cellData = append(cellData, []string{tag})
	}
	tabular.Display(header, cellData, false, -1)
	log.Println()
}

func (rg *ReportGenerator) writeUnparsableSQL(unparsableSQL []UnparsableSQL) {
	if len(unparsableSQL) > 0 {
		log.Println("Unparsable SQL Statements:")
		header := []string{"File", "ID", "SQL"}
		var cellData [][]string
		for _, sql := range unparsableSQL {
			cellData = append(cellData, []string{
				filepath.Base(sql.FilePath),
				sql.StmtID,
				sql.SQL,
			})
		}
		tabular.Display(header, cellData, false, -1)
		log.Printf("%d statements failed", len(unparsableSQL))
		log.Println()
	}
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
