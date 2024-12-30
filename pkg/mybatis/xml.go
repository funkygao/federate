package mybatis

import (
	"github.com/beevik/etree"
)

type XMLAnalyzer struct {
	Namespaces         map[string]int
	DynamicSQLElements map[string]int
}

func NewXMLAnalyzer() *XMLAnalyzer {
	return &XMLAnalyzer{
		Namespaces:         make(map[string]int),
		DynamicSQLElements: make(map[string]int),
	}
}

func (xa *XMLAnalyzer) Analyze(root *etree.Element) {
	xa.analyzeNamespace(root)
	for _, elem := range root.ChildElements() {
		xa.analyzeDynamicSQLElements(elem)
	}
}

func (xa *XMLAnalyzer) analyzeNamespace(root *etree.Element) {
	namespace := root.SelectAttrValue("namespace", "")
	xa.Namespaces[namespace]++
}

func (xa *XMLAnalyzer) analyzeDynamicSQLElements(elem *etree.Element) {
	dynamicElements := []string{"if", "choose", "when", "otherwise", "foreach"}
	for _, tag := range dynamicElements {
		xa.DynamicSQLElements[tag] += len(elem.FindElements(".//" + tag))
	}
}
