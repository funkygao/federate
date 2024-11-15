package spring

import (
	"log"
	"path/filepath"
	"strings"

	"github.com/beevik/etree"
)

const (
	logPrefix       = "%-13s"
	examiningPrefix = "  %-11s"
)

func (m *manager) SearchBean(springXmlPath string, beanId string) (bool, string) {
	log.Printf(logPrefix+"from: %s", "Starting search", springXmlPath)
	return m.searchBeanInFile(springXmlPath, beanId, make(map[string]bool))
}

func (m *manager) searchBeanInFile(filePath string, beanId string, visitedFiles map[string]bool) (bool, string) {
	if visitedFiles[filePath] {
		return false, ""
	}
	visitedFiles[filePath] = true

	// Handle wildcard in file path
	matches, err := filepath.Glob(filePath)
	if err != nil {
		log.Printf("Error expanding wildcard in path %s: %v", filePath, err)
		return false, ""
	}

	if len(matches) > 1 || strings.Contains(filePath, "*") {
		log.Printf(logPrefix+"%s", "Searching in", filePath)
	}

	for _, match := range matches {
		if len(matches) > 1 || strings.Contains(filePath, "*") {
			log.Printf(examiningPrefix+"%s", "Examining", match)
		} else {
			log.Printf(logPrefix+"%s", "Searching in", match)
		}

		doc := etree.NewDocument()
		if err := doc.ReadFromFile(match); err != nil {
			log.Printf("Error reading file %s: %v", match, err)
			continue
		}

		root := doc.Root()

		// Search for bean definitions
		for _, elem := range root.FindElements(".//*") {
			if isBeanElement(elem) {
				id := getBeanId(elem)
				if id == beanId {
					return true, match
				}
			}
		}

		// Search in imported files
		for _, imp := range root.FindElements("//import") {
			resource := imp.SelectAttrValue("resource", "")
			if resource != "" {
				importedPath := filepath.Join(filepath.Dir(match), resource)
				log.Printf(logPrefix+"%s", "Following", importedPath)
				found, foundFile := m.searchBeanInFile(importedPath, beanId, visitedFiles)
				if found {
					return true, foundFile
				}
			}
		}
	}

	return false, ""
}

func isBeanElement(elem *etree.Element) bool {
	isBean := elem.Tag == "bean" ||
		strings.HasSuffix(elem.Tag, ":bean") ||
		elem.Tag == "jmq:producer" ||
		elem.Tag == "jmq:consumer" ||
		elem.Tag == "jmq:transport" ||
		elem.Tag == "map" || // handle <util:map>
		elem.Tag == "list" || // handle <util:list>
		elem.Tag == "jsf:consumer" ||
		elem.Tag == "jsf:provider" ||
		elem.Tag == "dubbo:reference" ||
		elem.Tag == "dubbo:service"

	return isBean
}

func getBeanId(elem *etree.Element) string {
	id := elem.SelectAttrValue("id", "")
	if id == "" {
		id = elem.SelectAttrValue("name", "")
	}
	return id
}
