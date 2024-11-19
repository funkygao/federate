package spring

import (
	"log"
	"strings"

	"federate/pkg/util"
	"github.com/beevik/etree"
)

type BeanInfo struct {
	Bean     *etree.Element
	Value    string
	FileName string
}

type UpdateMap map[string]map[string]string // componentName:oldRef:newRef

func (um UpdateMap) RuleByFileName(fileName string) map[string]string {
	for componentName := range um {
		if strings.Contains(fileName, "/"+componentName+"/") {
			return um[componentName]
		}
	}
	return nil
}

type SpringManager interface {
	ListBeans(springXmlPath string, query Query) []BeanInfo

	UpdateBeans(springXmlPath string, query Query, updateMap UpdateMap) error
}

type manager struct {
	verbose bool
}

func New(verbose bool) SpringManager {
	return &manager{verbose: verbose}
}

func (m *manager) ListBeans(springXmlPath string, query Query) []BeanInfo {
	log.Printf("Starting from %s", springXmlPath)
	var beanInfos []BeanInfo
	processor := func(elem *etree.Element, beanInfo BeanInfo) (bool, error) {
		beanInfos = append(beanInfos, beanInfo)
		return false, nil
	}

	err := m.processBeansInFile(springXmlPath, query, make(map[string]bool), processor)
	if err != nil {
		log.Printf("Error processing beans: %v", err)
	}

	return beanInfos
}

func (m *manager) UpdateBeans(springXmlPath string, query Query, updateMap UpdateMap) error {
	processor := func(elem *etree.Element, beanInfo BeanInfo) (bool, error) {
		updates := updateMap.RuleByFileName(beanInfo.FileName)
		if updates != nil {
			if newValue, ok := updates[beanInfo.Value]; ok {
				if util.UpdateXmlElement(elem, "ref", newValue) {
					return true, nil
				}
			}
		}
		return false, nil
	}

	return m.processBeansInFile(springXmlPath, query, make(map[string]bool), processor)
}
