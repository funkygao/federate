package spring

import (
	"log"

	"github.com/beevik/etree"
)

const (
	logPrefix       = "%-13s"
	examiningPrefix = "  %-11s"
)

func (m *manager) ListBeans(springXmlPath string, searchType SearchType) []BeanInfo {
	log.Printf("Starting from %s", springXmlPath)
	var beanInfos []BeanInfo
	processor := func(elem *etree.Element, beanInfo BeanInfo) (bool, error) {
		beanInfos = append(beanInfos, beanInfo)
		return false, nil
	}

	err := m.processBeansInFile(springXmlPath, searchType, make(map[string]bool), processor)
	if err != nil {
		log.Printf("Error processing beans: %v", err)
	}

	return beanInfos
}

func (m *manager) ChangeBeans(springXmlPath string, searchType SearchType, updateMap map[string]string) error {
	processor := func(elem *etree.Element, beanInfo BeanInfo) (bool, error) {
		if newValue, ok := updateMap[beanInfo.Identifier]; ok {
			switch searchType {
			case SearchByRef:
				if elem.SelectAttr("ref") != nil {
					elem.RemoveAttr("ref")
					elem.CreateAttr("ref", newValue)
				}
			}
			return true, nil
		}
		return false, nil
	}

	return m.processBeansInFile(springXmlPath, searchType, make(map[string]bool), processor)
}
