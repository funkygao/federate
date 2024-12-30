package mybatis

type Analyzer struct {
	XMLAnalyzer     *XMLAnalyzer
	SQLAnalyzer     *SQLAnalyzer
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
	err := a.XMLAnalyzer.ReadFile(filePath)
	if err != nil {
		return err
	}

	a.XMLAnalyzer.Analyze()

	root := a.XMLAnalyzer.GetRoot()
	if root == nil {
		return nil // 不是 MyBatis mapper 文件
	}

	fragments, err := a.XMLAnalyzer.ExtractSQLFragments()
	if err != nil {
		return err
	}

	for _, elem := range root.ChildElements() {
		switch elem.Tag {
		case "select", "insert", "update", "delete":
			sql := a.XMLAnalyzer.extractSQL(elem)
			id := elem.SelectAttrValue("id", "")
			a.SQLAnalyzer.AnalyzeStmt(root, filePath, id, sql, fragments)

		default:
			a.SQLAnalyzer.IgnoreTag(elem.Tag)
		}
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.XMLAnalyzer, a.SQLAnalyzer)
}
