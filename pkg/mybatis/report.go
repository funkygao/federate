package mybatis

import (
	"fmt"
	"strings"

	"federate/pkg/primitive"
)

type ReportGenerator struct{}

func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{}
}

func (rg *ReportGenerator) Generate(xa *XMLAnalyzer, sa *SQLAnalyzer) string {
	var report strings.Builder

	report.WriteString("MyBatis Mapper Analysis Report\n")
	report.WriteString("==============================\n\n")

	rg.writeIgnoredTags(&report, sa.IgnoredTags)
	rg.writeNamespaces(&report, xa.Namespaces)
	rg.writeSQLTypes(&report, sa.SQLTypes)
	rg.writeMostUsedTables(&report, sa.Tables)
	rg.writeMostUsedFields(&report, sa.Fields)
	rg.writeComplexityMetrics(&report, sa)
	rg.writeAggregationFunctions(&report, sa.AggregationFuncs)
	rg.writeDynamicSQLElements(&report, xa.DynamicSQLElements)

	return report.String()
}

func (rg *ReportGenerator) writeIgnoredTags(report *strings.Builder, ignored *primitive.StringSet) {
	report.WriteString("Ignored Tags:\n")
	for _, tag := range ignored.Values() {
		report.WriteString(fmt.Sprintf("  <%s>\n", tag))
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeNamespaces(report *strings.Builder, namespaces map[string]int) {
	report.WriteString("Namespaces:\n")
	for ns, count := range namespaces {
		report.WriteString(fmt.Sprintf("  %s: %d\n", ns, count))
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeSQLTypes(report *strings.Builder, sqlTypes map[string]int) {
	report.WriteString("SQL Types:\n")
	for sqlType, count := range sqlTypes {
		report.WriteString(fmt.Sprintf("  %s: %d\n", sqlType, count))
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeMostUsedTables(report *strings.Builder, tables map[string]int) {
	report.WriteString("Most Used Tables:\n")
	for table, count := range tables {
		if count > 1 {
			report.WriteString(fmt.Sprintf("  %s: %d\n", table, count))
		}
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeMostUsedFields(report *strings.Builder, fields map[string]int) {
	report.WriteString("Most Used Fields:\n")
	for field, count := range fields {
		if count > 2 {
			report.WriteString(fmt.Sprintf("  %s: %d\n", field, count))
		}
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeComplexityMetrics(report *strings.Builder, sa *SQLAnalyzer) {
	report.WriteString("Complexity Metrics:\n")
	report.WriteString(fmt.Sprintf("  Complex Queries: %d\n", sa.ComplexQueries))
	report.WriteString(fmt.Sprintf("  Join Operations: %d\n", sa.JoinOperations))
	report.WriteString(fmt.Sprintf("  Subqueries: %d\n", sa.SubQueries))
	report.WriteString(fmt.Sprintf("  Distinct Queries: %d\n", sa.DistinctQueries))
	report.WriteString(fmt.Sprintf("  Order By Operations: %d\n", sa.OrderByOperations))
	report.WriteString(fmt.Sprintf("  Limit Operations: %d\n", sa.LimitOperations))
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeAggregationFunctions(report *strings.Builder, aggFuncs map[string]int) {
	report.WriteString("Aggregation Functions:\n")
	for func_, count := range aggFuncs {
		report.WriteString(fmt.Sprintf("  %s: %d\n", func_, count))
	}
	report.WriteString("\n")
}

func (rg *ReportGenerator) writeDynamicSQLElements(report *strings.Builder, elements map[string]int) {
	report.WriteString("Dynamic SQL Elements:\n")
	for element, count := range elements {
		report.WriteString(fmt.Sprintf("  %s: %d\n", element, count))
	}
	report.WriteString("\n")
}
