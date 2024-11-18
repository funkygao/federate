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
	refTransformMap := make(map[string]string)
	for _, c := range m.m.Components {
		if err := javast.TransformService(c); err != nil {
			return err
		}

		for k, v := range c.Transform.ServiceBeanRefMap() {
			refTransformMap[k] = v
		}
	}

	log.Printf("Update corresponding xml ref rule: %+v", refTransformMap)
	springXmlPath := filepath.Join(m.m.TargetResourceDir(), "federated/spring.xml")
	springMgr := spring.New()
	return springMgr.ChangeBeans(springXmlPath, spring.SearchByRef, refTransformMap)
}
