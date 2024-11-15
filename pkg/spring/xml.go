package spring

import (
	"log"
	"path/filepath"
	"strings"

	"github.com/beevik/etree"
)

func (m *manager) SearchBean(springXmlPath string, beanId string) (bool, string) {
	log.Printf("Starting search %s from: %s", beanId, springXmlPath)
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
		log.Printf("Searching in file: %s", filePath)
	}

	for _, match := range matches {
		if len(matches) > 1 || strings.Contains(filePath, "*") {
			log.Printf("   Examining file: %s", match)
		} else {
			log.Printf("Searching in file: %s", match)
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
					printBeanInfo(elem, match)
					return true, match
				}
			}
		}

		// Search in imported files
		for _, imp := range root.FindElements("//import") {
			resource := imp.SelectAttrValue("resource", "")
			if resource != "" {
				importedPath := filepath.Join(filepath.Dir(match), resource)
				log.Printf("Following import:  %s", importedPath)
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
	return elem.Tag == "bean" ||
		strings.HasSuffix(elem.Tag, ":bean") ||
		elem.Tag == "jmq:producer" ||
		elem.Tag == "jmq:consumer" ||
		elem.Tag == "jmq:transport" ||
		elem.Tag == "util:list" ||
		elem.Tag == "util:set" ||
		elem.Tag == "util:map" ||
		elem.Tag == "jsf:consumer" ||
		elem.Tag == "dubbo:reference" ||
		elem.Tag == "dubbo:service"
}

func getBeanId(elem *etree.Element) string {
	id := elem.SelectAttrValue("id", "")
	if id == "" {
		id = elem.SelectAttrValue("name", "")
	}
	return id
}

func printBeanInfo(elem *etree.Element, filePath string) {
	log.Printf("Bean details:")
	log.Printf("File: %s", filePath)
	log.Printf("Element: %s", elem.Tag)
	log.Printf("ID: %s", getBeanId(elem))
	log.Printf("Class: %s", elem.SelectAttrValue("class", "N/A"))

	// Print additional attributes
	for _, attr := range elem.Attr {
		if attr.Key != "id" && attr.Key != "name" && attr.Key != "class" {
			log.Printf("Attribute %s: %s", attr.Key, attr.Value)
		}
	}

	// Print child elements (e.g., properties, constructor-args)
	for _, child := range elem.ChildElements() {
		log.Printf("Child element: %s", child.Tag)
		for _, attr := range child.Attr {
			log.Printf("  %s: %s", attr.Key, attr.Value)
		}
	}
}
