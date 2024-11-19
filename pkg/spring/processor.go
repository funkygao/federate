package spring

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/beevik/etree"
)

const (
	logPrefix       = "%-13s"
	examiningPrefix = "  %-11s"
)

func (m *manager) processBeansInFile(filePath string, query Query, visitedFiles map[string]bool,
	processor func(elem *etree.Element, beanInfo BeanInfo) (bool, error)) error {
	if visitedFiles[filePath] {
		return nil
	}
	visitedFiles[filePath] = true

	matches, err := filepath.Glob(filePath)
	if err != nil {
		return fmt.Errorf("error expanding wildcard in path %s: %v", filePath, err)
	}

	for _, match := range matches {
		if m.verbose && (len(matches) > 1 || strings.Contains(filePath, "*")) {
			log.Printf(examiningPrefix+"%s", "Examining", match)
		}

		doc := etree.NewDocument()
		if err := doc.ReadFromFile(match); err != nil {
			return fmt.Errorf("error reading file %s: %v", match, err)
		}

		root := doc.Root()
		changed := false

		for _, elem := range root.FindElements(".//*") {
			var (
				beanInfo  BeanInfo
				beanFound = false
			)

			if query.predicate == nil || query.predicate(elem) {
				if value, found := m.getValueByQuery(elem, query); found {
					beanFound = true
					beanInfo = BeanInfo{
						Bean:     elem,
						Value:    value,
						FileName: match,
					}
				}
			}

			if beanFound {
				// 找到该bean信息，才交给 processor 处理
				elemChanged, err := processor(elem, beanInfo)
				if err != nil {
					return err
				}
				changed = changed || elemChanged
			}
		}

		// If changes were made, save the file
		if changed {
			if err := doc.WriteToFile(match); err != nil {
				return fmt.Errorf("error writing changes to file %s: %v", match, err)
			}
			log.Printf("Updated %s", match)
		}

		// Recursively process imported files
		for _, importElem := range root.FindElements("//import") {
			resourceAttr := importElem.SelectAttrValue("resource", "")
			resourceAttr = strings.TrimPrefix(resourceAttr, "classpath:") // 移除 resource 属性中的 classpath: 前缀
			if resourceAttr != "" {
				importedPath := filepath.Join(filepath.Dir(match), resourceAttr)
				if m.verbose {
					log.Printf(logPrefix+"%s", "Following", importedPath)
				}
				if err := m.processBeansInFile(importedPath, query, visitedFiles, processor); err != nil {
					return err
				}
			} else {
				log.Printf("Invalid import: %s", importElem.SelectAttrValue("resource", ""))
			}
		}
	}

	return nil
}

func (m *manager) getValueByQuery(elem *etree.Element, q Query) (value string, found bool) {
	for _, attr := range q.attributes {
		if val := elem.SelectAttrValue(attr, ""); val != "" {
			if q.queryString == "" || val == q.queryString {
				// 如果 queryString 为空，我们不进行精确匹配，而是返回第一个非空值
				return val, true
			}
		}
	}

	for tag, attrs := range q.tags {
		if elem.Tag == tag {
			for _, attr := range attrs {
				if val := elem.SelectAttrValue(attr, ""); val != "" {
					if q.queryString == "" || val == q.queryString {
						return val, true
					}
				}
			}
		}
	}

	// not found
	return "", false
}
