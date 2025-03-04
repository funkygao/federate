package merge

import (
	"log"

	"federate/pkg/javast"
	"federate/pkg/manifest"
	"federate/pkg/spring"
)

// 处理 @Service/@Component 的 Bean 创建
type ServiceManager struct {
	m *manifest.Manifest
}

func NewServiceManager(m *manifest.Manifest) Reconciler {
	return &ServiceManager{m: m}
}

func (m *ServiceManager) Name() string {
	return "Transform @Service/@Component for directive 'transform.service' and Update reference XML"
}

func (m *ServiceManager) Reconcile() error {
	// pass 1: 通过 javast 修改源代码
	refTransformMap := make(map[string]map[string]string)
	for _, c := range m.m.Components {
		javast.BacklogTransformComponentBean(c)

		if refTransformMap[c.Name] == nil {
			refTransformMap[c.Name] = make(map[string]string)
		}

		for k, v := range c.Transform.ServiceBeanRefMap() {
			refTransformMap[c.Name][k] = v
		}
	}

	for c, rule := range refTransformMap {
		if len(rule) > 0 {
			log.Printf("[%s] XML Ref Plan: %v", c, rule)
		}
	}

	// pass 2: 修改相应的 xml ref
	springMgr := spring.New(false)
	return springMgr.UpdateBeans(m.m.SpringXmlPath(), spring.QueryRef(), refTransformMap)
}
