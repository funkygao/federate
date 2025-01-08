package mybatis

import "log"

const promtContext = `
# Apache MyBatis Mapper XML File Analysis Report

## Background

This report provides a thorough analysis of the MyBatis Mapper XML files within my project. MyBatis is a Java persistence framework that employs XML configuration files, called Mapper XML files, to establish mappings between SQL statements and Java objects. These files are crucial for efficient database interactions and maintaining code readability.

By analyzing the XML mapper files, we can uncover the primary business operations and processes, as they define how data is accessed, modified, and utilized to support key business functions. This analysis allows us to gain insights into the system's data physical model and understand the underlying logic that drives the business.

## Your role

As a SQL Analysis Expert and Business Insights Consultant, your responsibilities include:

- Conducting a detailed analysis of the MyBatis Mapper XML file analysis report
- Identifying the main business operations and processes based on the SQL statements and their interactions with the database
- Assessing the current state and health of the business system by understanding how data is used to support business objectives
- Providing expert recommendations to optimize the alignment between database operations and business requirements
- Offering insights into potential improvements in business processes based on the analysis of data usage patterns
- Suggesting areas for further investigation or clarification if the report does not provide sufficient information for a comprehensive understanding of the business operations

## Expected Outputs

- Initial Observations: Begin by describing what you have observed in the analysis report. Highlight the key findings related to SQL types, table usage, join analysis, aggregation functions, and any notable patterns or trends that stand out. This will provide a foundation for further insights and recommendations
- Clear identification of the primary business operations and processes within the system
- Insights into how data is utilized to support these business functions and any potential gaps or inefficiencies
- Practical recommendations for enhancing the effectiveness and efficiency of business operations through database optimizations
- Strategic suggestions for leveraging the analysis findings to drive business growth and improve overall system performance
- Guidance on potential areas for future analysis or enhancements to gain deeper insights into the business operations

Please provide further insights and recommendations based on this analysis report.
`

var (
	Verbosity           int
	TopK                int
	SimilarityThreshold float64
	GeneratePrompt      bool
	ShowIndexRecommend  bool
	ShowBatchOps        bool
	ShowSimilarity      bool
)

type Analyzer struct {
	aggregator      *Aggregator
	reportGenerator *ReportGenerator
	mapperBuilders  map[string]*XMLMapperBuilder
}

func NewAnalyzer(ignoredFields []string) *Analyzer {
	return &Analyzer{
		aggregator:      NewAggregator(ignoredFields),
		reportGenerator: NewReportGenerator(),
		mapperBuilders:  make(map[string]*XMLMapperBuilder),
	}
}

func (a *Analyzer) AnalyzeFiles(files []string) {
	// prepare sql fragments
	for _, file := range files {
		if err := a.prepareFile(file); err != nil {
			log.Printf("%s %v", file, err)
		}
	}

	// analyze CRUD
	for _, file := range files {
		if err := a.analyzeFile(file); err != nil {
			log.Printf("%s %v", file, err)
		}
	}

	a.aggregator.Aggregate()
}

func (a *Analyzer) prepareFile(filePath string) error {
	xml := NewXMLMapperBuilder(filePath)
	if err := xml.Prepare(); err != nil && err != ErrNotMapperXML {
		return err
	}

	a.mapperBuilders[filePath] = xml
	return nil
}

func (a *Analyzer) analyzeFile(filePath string) error {
	// 解析 XML
	xml := a.mapperBuilders[filePath]
	if xml == nil {
		// not a mapper XML, e,g. pom.xml
		return nil
	}

	stmts, err := xml.Parse()
	if err != nil {
		if err == ErrNotMapperXML {
			return nil
		}
		return err
	}

	// <include refid=""/> 没有找到被引用 sql fragment
	a.aggregator.UnknownFragments[filePath] = xml.UnknownFragments

	// 解析 SQL AST
	for _, stmt := range stmts {
		a.aggregator.OnStmt(*stmt)
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.reportGenerator.Generate(a.aggregator)
}
