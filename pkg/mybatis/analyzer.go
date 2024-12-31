package mybatis

type Analyzer struct {
	SQLAnalyzer     *SQLAnalyzer
	ReportGenerator *ReportGenerator
}

func NewAnalyzer() *Analyzer {
	return &Analyzer{
		SQLAnalyzer:     NewSQLAnalyzer(),
		ReportGenerator: NewReportGenerator(),
	}
}

func (a *Analyzer) AnalyzeFile(filePath string) error {
	builder := NewXMLMapperBuilder(filePath)
	stmts, err := builder.Parse()
	if err != nil {
		if err == ErrNotMapperXML {
			return nil
		}
		return err
	}

	for _, stmt := range stmts {
		a.SQLAnalyzer.AnalyzeStmt(*stmt)
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.SQLAnalyzer)
}
