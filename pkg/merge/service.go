package merge

import (
	"federate/pkg/javast"
	"federate/pkg/manifest"
)

// 处理 @Service 的 Bean 创建
type ServiceManager struct {
	m *manifest.Manifest
}

func NewServiceManager(m *manifest.Manifest) *ServiceManager {
	return &ServiceManager{m: m}
}

func (m *ServiceManager) Reconcile() error {
	for _, c := range m.m.Components {
		if err := javast.TransformService(c); err != nil {
			return err
		}
	}
	return nil
}
