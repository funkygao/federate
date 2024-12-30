package mybatis

import (
	"strings"

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

func (xa *XMLAnalyzer) ReadFile(filePath string) error {
	xa.Doc = etree.NewDocument()
	return xa.Doc.ReadFromFile(filePath)
}

func (xa *XMLAnalyzer) ReadFromString(xmlContent string) error {
	xa.Doc = etree.NewDocument()
	return xa.Doc.ReadFromString(xmlContent)
}

func (xa *XMLAnalyzer) Analyze() {
	root := xa.Doc.SelectElement(RootTag)
	if root == nil {
		return // 不是 MyBatis mapper 文件
	}

	xa.root = root
	xa.analyzeNamespace(root)
	xa.analyzeDynamicSQLElements(root)
}

func (xa *XMLAnalyzer) ExtractSQLFragments() (SQLFragments, error) {
	fragments := make(SQLFragments)
	for _, elem := range xa.root.ChildElements() {
		if elem.Tag == "sql" {
			id := elem.SelectAttrValue("id", "")
			if id != "" {
				fragments[id] = xa.extractRawSQL(elem)
			}
		}
	}

	return fragments, nil
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

func (xa *XMLAnalyzer) GetRoot() *etree.Element {
	return xa.root
}

func (xa *XMLAnalyzer) extractRawSQL(elem *etree.Element) string {
	var sql strings.Builder
	xa.extractSQLRecursive(elem, &sql)
	return strings.TrimSpace(sql.String())
}

func (xa *XMLAnalyzer) extractSQLRecursive(elem *etree.Element, sql *strings.Builder) {
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			sql.WriteString(v.Data)
		case *etree.Element:
			if v.Tag == "![CDATA[" {
				sql.WriteString(v.Text())
			} else {
				sql.WriteString("<" + v.Tag)
				for _, attr := range v.Attr {
					sql.WriteString(" " + attr.Key + "=\"" + attr.Value + "\"")
				}
				sql.WriteString(">")
				xa.extractSQLRecursive(v, sql)
				sql.WriteString("</" + v.Tag + ">")
			}
		}
	}
}
