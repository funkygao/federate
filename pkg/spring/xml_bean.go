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

func (m *manager) ListBeans(springXmlPath string, searchType SearchType) []BeanInfo {
	log.Printf("Starting from %s", springXmlPath)
	beanInfos := m.listBeansInFile(springXmlPath, searchType, make(map[string]bool))

	if m.showUnregistered && len(m.unregisteredTags) > 0 {
		for tag := range m.unregisteredTags {
			log.Printf("Unregisted tag: %s", tag)
		}
	}
	return beanInfos
}

func (m *manager) listBeansInFile(filePath string, searchType SearchType, visitedFiles map[string]bool) []BeanInfo {
	if visitedFiles[filePath] {
		return nil
	}
	visitedFiles[filePath] = true

	var beanInfos []BeanInfo

	matches, err := filepath.Glob(filePath)
	if err != nil {
		log.Printf("Error expanding wildcard in path %s: %v", filePath, err)
		return nil
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

		// Search for bean definitions or refs
		for _, elem := range root.FindElements(".//*") {
			switch searchType {
			case SearchByID:
				if m.isBeanElement(elem) {
					if id := getBeanId(elem); id != "" {
						beanInfos = append(beanInfos, BeanInfo{Identifier: id, FileName: match})
					}
				}

			case SearchByRef:
				if ref := getRefValue(elem); ref != "" {
					beanInfos = append(beanInfos, BeanInfo{Identifier: ref, FileName: match})
				}
			}
		}

		// Recursively search in imported files
		for _, imp := range root.FindElements("//import") {
			resource := imp.SelectAttrValue("resource", "")
			if resource != "" {
				importedPath := filepath.Join(filepath.Dir(match), resource)
				log.Printf(logPrefix+"%s", "Following", importedPath)
				beanInfos = append(beanInfos, m.listBeansInFile(importedPath, searchType, visitedFiles)...)
			}
		}
	}

	return beanInfos
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
