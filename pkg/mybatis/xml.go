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
	xa.analyzeDynamicSQLElements(root)
}

func (xa *XMLAnalyzer) analyzeNamespace(root *etree.Element) {
	namespace := root.SelectAttrValue("namespace", "")
	xa.Namespaces[namespace]++
}

func (xa *XMLAnalyzer) analyzeDynamicSQLElements(elem *etree.Element) {
	// TODO foreach is under root?
	dynamicElements := []string{"if", "choose", "when", "otherwise", "foreach"}
	for _, tag := range dynamicElements {
		if elem.FindElements(".//"+tag) != nil {
			xa.DynamicSQLElements[tag]++
		}
	}
}
