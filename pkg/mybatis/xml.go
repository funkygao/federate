package mybatis

import (
	"github.com/beevik/etree"
)

type XMLAnalyzer struct {
	Namespaces         map[string]int
	DynamicSQLElements map[string]int
	Doc                *etree.Document
	root               *etree.Element
}

func NewXMLAnalyzer() *XMLAnalyzer {
	return &XMLAnalyzer{
		Namespaces:         make(map[string]int),
		DynamicSQLElements: make(map[string]int),
	}
}

func (xa *XMLAnalyzer) GetRoot() *etree.Element {
	return xa.root
}

func (xa *XMLAnalyzer) AnalyzeFile(filePath string) error {
	xa.Doc = etree.NewDocument()
	if err := xa.Doc.ReadFromFile(filePath); err != nil {
		return err
	}

	xa.analyze()
	return nil
}

func (xa *XMLAnalyzer) AnalyzeString(xmlContent string) error {
	xa.Doc = etree.NewDocument()
	if err := xa.Doc.ReadFromString(xmlContent); err != nil {
		return err
	}

	xa.analyze()
	return nil
}

func (xa *XMLAnalyzer) analyze() {
	root := xa.Doc.SelectElement(RootTag)
	if root == nil {
		return // 不是 MyBatis mapper 文件
	}

	xa.root = root
	xa.analyzeNamespace(root)
	xa.analyzeDynamicSQLElements(root)
}

func (xa *XMLAnalyzer) analyzeNamespace(root *etree.Element) {
	namespace := root.SelectAttrValue("namespace", "")
	xa.Namespaces[namespace]++
}

func (xa *XMLAnalyzer) analyzeDynamicSQLElements(root *etree.Element) {
	dynamicElements := []string{"if", "choose", "when", "otherwise", "foreach"}

	for _, elem := range root.ChildElements() {
		if elem.Tag == "select" || elem.Tag == "update" || elem.Tag == "insert" || elem.Tag == "delete" {
			xa.countDynamicElements(elem, dynamicElements)
		}
	}
}

func (xa *XMLAnalyzer) countDynamicElements(elem *etree.Element, dynamicElements []string) {
	for _, child := range elem.ChildElements() {
		if contains(dynamicElements, child.Tag) {
			xa.DynamicSQLElements[child.Tag]++
		}
		xa.countDynamicElements(child, dynamicElements)
	}
}

func contains(slice []string, item string) bool {
	for _, a := range slice {
		if a == item {
			return true
		}
	}
	return false
}
