package spring

import (
	"log"
	"strings"

	"github.com/beevik/etree"
)

type SearchType int

const (
	logPrefix       = "%-13s"
	examiningPrefix = "  %-11s"

	SearchByID SearchType = iota
	SearchByRef
)

type BeanInfo struct {
	// Identifier 可能是 bean 的 ID/name（当 Type 为 SearchByID 时），
	// 或者是 ref 值（当 Type 为 SearchByRef 时）
	Identifier string

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
	ListBeans(springXmlPath string, searchType SearchType) []BeanInfo

	ChangeBeans(springXmlPath string, searchType SearchType, updateMap UpdateMap) error
}

type manager struct {
	verbose bool

	beanFullTags map[string]struct{}

	showUnregistered bool
	unregisteredTags map[string]struct{}
}

func New(verbose bool) SpringManager {
	return &manager{
		verbose: verbose,
		beanFullTags: map[string]struct{}{
			"bean":               struct{}{},
			"util:map":           struct{}{},
			"util:list":          struct{}{},
			"laf-config:manager": struct{}{},
			"jmq:producer":       struct{}{},
			"jmq:consumer":       struct{}{},
			"jmq:transport":      struct{}{},
			"jsf:consumer":       struct{}{},
			"jsf:consumerGroup":  struct{}{},
			"jsf:provider":       struct{}{},
			"jsf:filter":         struct{}{},
			"jsf:server":         struct{}{},
			"jsf:registry":       struct{}{},
			"dubbo:reference":    struct{}{},
			"dubbo:service":      struct{}{},
		},
		showUnregistered: false,
		unregisteredTags: make(map[string]struct{}),
	}
}

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

func (m *manager) ChangeBeans(springXmlPath string, searchType SearchType, updateMap UpdateMap) error {
	if searchType != SearchByRef {
		// do nothing
		return nil
	}

	processor := func(elem *etree.Element, beanInfo BeanInfo) (bool, error) {
		updates := updateMap.RuleByFileName(beanInfo.FileName)
		if updates != nil {
			if newValue, ok := updates[beanInfo.Identifier]; ok {
				if elem.SelectAttr("ref") != nil {
					elem.RemoveAttr("ref")
					elem.CreateAttr("ref", newValue)
					return true, nil
				}
			}
		}
		return false, nil
	}

	return m.processBeansInFile(springXmlPath, searchType, make(map[string]bool), processor)
}
