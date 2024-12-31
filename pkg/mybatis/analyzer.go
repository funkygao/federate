package mybatis

import (
	"github.com/xwb1989/sqlparser"
	"log"
)

type Analyzer struct {
	ok   int
	fail int

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
	if err := builder.Parse(); err != nil {
		return err
	}

	for id, stmt := range builder.Statements {
		_, err := sqlparser.Parse(stmt.ParseableSQL)
		if err != nil {
			a.fail++
			log.Printf("%s %s\n%v", id, stmt.ParseableSQL, err)
			log.Println()
		} else {
			a.ok++
		}
	}

	log.Printf("ok: %d, fail:%d", a.ok, a.fail)

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.SQLAnalyzer)
}
