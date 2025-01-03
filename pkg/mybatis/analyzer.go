package mybatis

import "log"

type Analyzer struct {
	SQLAnalyzer     *SQLAnalyzer
	ReportGenerator *ReportGenerator

	mapperBuilders map[string]*XMLMapperBuilder
}

func NewAnalyzer(ignoredFields []string) *Analyzer {
	db, err := NewDB(DbFile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	if err := db.InitTables(); err != nil {
		log.Fatalf("%v", err)
	}

	return &Analyzer{
		SQLAnalyzer:     NewSQLAnalyzer(ignoredFields, db),
		ReportGenerator: NewReportGenerator(),
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
	xml := a.mapperBuilders[filePath]
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
