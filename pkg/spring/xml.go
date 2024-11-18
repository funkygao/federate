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

func (m *manager) SearchBean(springXmlPath string, beanId string) (found bool, path string) {
	log.Printf("Starting from %s", springXmlPath)
	found, path = m.searchBeanInFile(springXmlPath, beanId, make(map[string]bool))
	if m.showUnregistered && len(m.unregisteredTags) > 0 {
		for tag := range m.unregisteredTags {
			log.Printf("Unregisted tag: %s", tag)
		}
	}
	return
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

	for _, match := range matches {
		if len(matches) > 1 || strings.Contains(filePath, "*") {
			log.Printf(examiningPrefix+"%s", "Examining", match)
		}

		doc := etree.NewDocument()
		if err := doc.ReadFromFile(match); err != nil {
			log.Printf("Error reading file %s: %v", match, err)
			continue
		}

		root := doc.Root()

		// Search for bean definitions
		for _, elem := range root.FindElements(".//*") {
			if m.isBeanElement(elem) {
				if id := getBeanId(elem); id == beanId {
					return true, match
				}
			}
		}

		// Recursively search in imported files
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
