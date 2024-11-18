package merge

import (
	"log"
	"path/filepath"

	"federate/pkg/javast"
	"federate/pkg/manifest"
	"federate/pkg/spring"
)

// 处理 @Service 的 Bean 创建
type ServiceManager struct {
	m *manifest.Manifest
}

func NewServiceManager(m *manifest.Manifest) *ServiceManager {
	return &ServiceManager{m: m}
}

func (m *ServiceManager) Reconcile() error {
	refTransformMap := make(map[string]map[string]string)
	for _, c := range m.m.Components {
		if err := javast.TransformService(c); err != nil {
			return err
		}

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

	springXmlPath := filepath.Join(m.m.TargetResourceDir(), "federated/spring.xml")
	springMgr := spring.New(false)
	return springMgr.ChangeBeans(springXmlPath, spring.SearchByRef, refTransformMap)
}
