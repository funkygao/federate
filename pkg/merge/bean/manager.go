package bean

import (
	"strings"

	"federate/pkg/manifest"
)

type XmlBeanManager struct {
	m         *manifest.Manifest
	beanIdMap map[string][]BeanIdInfo // key 是 bean id，value 是 BeanIdInfo 列表
	plan      ReconcilePlan

	sameComponentFunc, otherComponentFunc func(componentName string, i BeanIdInfo) bool
}

// BeanIdInfo 结构体用于存储 bean id 的相关信息
type BeanIdInfo struct {
	BeanId         string
	ClassFullName  string   // Bean 类的全名
	ComponentName  string   // 组件名称
	SourceFilePath string   // 源文件路径
	TargetFilePath string   // 目标文件路径
	ParentPath     []string // 父元素路径,用于记录嵌套关系
}

func (b *BeanIdInfo) Nested() bool {
	return len(b.ParentPath) > 1 // `beans` is always the root parent
}

func (b *BeanIdInfo) SameId(that BeanIdInfo) bool {
	return b.BeanId == that.BeanId &&
		strings.Join(b.ParentPath, "/") == strings.Join(that.ParentPath, "/")
}

// NewXmlBeanManager 创建一个新的 XmlBeanManager
func NewXmlBeanManager(m *manifest.Manifest) *XmlBeanManager {
	return &XmlBeanManager{
		m:         m,
		beanIdMap: make(map[string][]BeanIdInfo),
		plan:      newReconcilePlan(m),
		sameComponentFunc: func(componentName string, i BeanIdInfo) bool {
			return componentName == i.ComponentName
		},
		otherComponentFunc: func(componentName string, i BeanIdInfo) bool {
			return componentName != i.ComponentName
		},
	}
}

func (m *XmlBeanManager) ComponentBeans(componentName string, predicate func(c string, i BeanIdInfo) bool) []BeanIdInfo {
	var beans []BeanIdInfo
	for _, infos := range m.beanIdMap {
		for _, info := range infos {
			if predicate(componentName, info) {
				beans = append(beans, info)
			}
		}
	}
	return beans
}

func (m *XmlBeanManager) ReconcilePlan() *ReconcilePlan {
	return &m.plan
}
