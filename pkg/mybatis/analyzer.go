package mybatis

import "log"

type Analyzer struct {
	SQLAnalyzer     *SQLAnalyzer
	ReportGenerator *ReportGenerator

	builders map[string]*XMLMapperBuilder
}

func NewAnalyzer(ignoredFields []string) *Analyzer {
	return &Analyzer{
		SQLAnalyzer:     NewSQLAnalyzer(ignoredFields),
		ReportGenerator: NewReportGenerator(),
		builders:        make(map[string]*XMLMapperBuilder),
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
}

func (a *Analyzer) prepareFile(filePath string) error {
	xml := NewXMLMapperBuilder(filePath)
	if err := xml.Prepare(); err != nil && err != ErrNotMapperXML {
		return err
	}

	a.builders[filePath] = xml
	return nil
}

func (a *Analyzer) analyzeFile(filePath string) error {
	xml := a.builders[filePath]
	if xml == nil {
		return nil
	}
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
