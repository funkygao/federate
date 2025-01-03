package mybatis

type Analyzer struct {
	SQLAnalyzer     *SQLAnalyzer
	ReportGenerator *ReportGenerator
}

func NewAnalyzer(ignoredFields []string) *Analyzer {
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

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.SQLAnalyzer)
}
