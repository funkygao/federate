package spring

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/beevik/etree"
)

type BeanProcessor func(elem *etree.Element, beanInfo BeanInfo) (bool, error)

func (m *manager) processBeansInFile(filePath string, searchType SearchType, visitedFiles map[string]bool, processor BeanProcessor) error {
	if visitedFiles[filePath] {
		return nil
	}
	visitedFiles[filePath] = true

	matches, err := filepath.Glob(filePath)
	if err != nil {
		return fmt.Errorf("error expanding wildcard in path %s: %v", filePath, err)
	}

	for _, match := range matches {
		if len(matches) > 1 || strings.Contains(filePath, "*") {
			log.Printf(examiningPrefix+"%s", "Examining", match)
		}

		doc := etree.NewDocument()
		if err := doc.ReadFromFile(match); err != nil {
			return fmt.Errorf("error reading file %s: %v", match, err)
		}

		root := doc.Root()
		changed := false

		// Process bean definitions or refs
		for _, elem := range root.FindElements(".//*") {
			var beanInfo BeanInfo
			switch searchType {
			case SearchByID:
				if m.isBeanElement(elem) {
					if id := getBeanId(elem); id != "" {
						beanInfo = BeanInfo{Identifier: id, FileName: match}
					}
				}
			case SearchByRef:
				if ref := getRefValue(elem); ref != "" {
					beanInfo = BeanInfo{Identifier: ref, FileName: match}
				}
			}

			if beanInfo.Identifier != "" {
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
		for _, imp := range root.FindElements("//import") {
			resource := imp.SelectAttrValue("resource", "")
			if resource != "" {
				importedPath := filepath.Join(filepath.Dir(match), resource)
				log.Printf(logPrefix+"%s", "Following", importedPath)
				if err := m.processBeansInFile(importedPath, searchType, visitedFiles, processor); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func getRefValue(elem *etree.Element) string {
	if ref := elem.SelectAttrValue("ref", ""); ref != "" {
		return ref
	}
	if elem.Tag == "ref" {
		return elem.SelectAttrValue("bean", "")
	}
	return ""
}

func (m *manager) isBeanElement(elem *etree.Element) bool {
	var fullTag string
	if elem.Space == "" {
		// <bean>
		fullTag = elem.Tag
	} else {
		fullTag = elem.Space + ":" + elem.Tag
	}

	_, present := m.beanFullTags[fullTag]
	if !present {
		m.unregisteredTags[fullTag] = struct{}{}
	}
	return present
}

func getBeanId(elem *etree.Element) string {
	id := elem.SelectAttrValue("id", "")
	if id == "" {
		id = elem.SelectAttrValue("name", "")
	}
	return id
}
