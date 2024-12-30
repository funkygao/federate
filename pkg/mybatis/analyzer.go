package mybatis

import (
	"strings"

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
			sql := extractSQL(elem)
			id := elem.SelectAttrValue("id", "")
			a.SQLAnalyzer.Analyze(filePath, id, sql)

		default:
			a.SQLAnalyzer.IgnoreTag(elem.Tag)
		}
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.XMLAnalyzer, a.SQLAnalyzer)
}

func extractSQL(elem *etree.Element) string {
	var sql strings.Builder
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			sql.WriteString(v.Data)
		case *etree.Element:
			if v.Tag == "![CDATA[" {
				sql.WriteString(v.Text())
			}
		}
	}
	return strings.TrimSpace(sql.String())
}
