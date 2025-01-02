package mybatis

type Analyzer struct {
	SQLAnalyzer     *SQLAnalyzer
	ReportGenerator *ReportGenerator
}

func NewAnalyzer(inVerbosity int, ignoredFields []string) *Analyzer {
	verbosity = inVerbosity

	return &Analyzer{
		SQLAnalyzer:     NewSQLAnalyzer(ignoredFields),
		ReportGenerator: NewReportGenerator(),
	}
}

func (a *Analyzer) AnalyzeFile(filePath string) error {
	xml := NewXMLMapperBuilder(filePath)
	stmts, err := xml.Parse()
	if err != nil {
		if err == ErrNotMapperXML {
			return nil
		}
		return err
	}

	a.SQLAnalyzer.Visit(filePath, xml.UnknownFragments)

	for _, stmt := range stmts {
		a.SQLAnalyzer.AnalyzeStmt(*stmt)
	}

	return nil
}

func (a *Analyzer) GenerateReport(topK int) {
	a.ReportGenerator.Generate(a.SQLAnalyzer, topK)
}
