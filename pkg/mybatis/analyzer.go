package mybatis

import (
	"github.com/beevik/etree"
)

type Analyzer struct {
	XMLAnalyzer *XMLAnalyzer
	SQLAnalyzer *SQLAnalyzer

	ReportGenerator *ReportGenerator
}

func NewAnalyzer() *Analyzer {
	return &Analyzer{
		XMLAnalyzer:     NewXMLAnalyzer(),
		SQLAnalyzer:     NewSQLAnalyzer(),
		ReportGenerator: NewReportGenerator(),
	}
}

func (a *Analyzer) AnalyzeFile(filePath string) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return err
	}

	root := doc.SelectElement("mapper")
	if root == nil {
		return nil // 不是 MyBatis mapper 文件
	}

	a.XMLAnalyzer.Analyze(root)
	for _, elem := range root.ChildElements() {
		switch elem.Tag {
		case "select", "insert", "update", "delete":
			a.SQLAnalyzer.Analyze(elem.Text())
		default:
			a.SQLAnalyzer.IgnoreTag(elem.Tag)
		}
	}

	return nil
}

func (a *Analyzer) GenerateReport() string {
	return a.ReportGenerator.Generate(a.XMLAnalyzer, a.SQLAnalyzer)
}
