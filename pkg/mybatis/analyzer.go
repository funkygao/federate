package mybatis

type Analyzer struct {
	XMLAnalyzer      *XMLAnalyzer
	SQLAnalyzer      *SQLAnalyzer
	MyBatisProcessor *MyBatisProcessor
	ReportGenerator  *ReportGenerator
}

func NewAnalyzer() *Analyzer {
	return &Analyzer{
		XMLAnalyzer:      NewXMLAnalyzer(),
		SQLAnalyzer:      NewSQLAnalyzer(),
		MyBatisProcessor: NewMyBatisProcessor(),
		ReportGenerator:  NewReportGenerator(),
	}
}

func (a *Analyzer) AnalyzeFile(filePath string) error {
	if err := a.XMLAnalyzer.AnalyzeFile(filePath); err != nil {
		return err
	}

	root := a.XMLAnalyzer.GetRoot()
	if root == nil {
		return nil // 不是 MyBatis mapper 文件
	}

	a.MyBatisProcessor.ExtractSQLFragments(root)

	for _, stmt := range root.ChildElements() {
		switch stmt.Tag {
		case "select", "insert", "update", "delete":
			_, preprocessedSQL, stmtID := a.MyBatisProcessor.PreprocessStmt(stmt)
			a.SQLAnalyzer.AnalyzeStmt(filePath, stmtID, preprocessedSQL)

		default:
			a.SQLAnalyzer.IgnoreTag(stmt.Tag)
		}
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.ReportGenerator.Generate(a.XMLAnalyzer, a.SQLAnalyzer)
}
