package merge

import (
	"federate/pkg/federated"
	"federate/pkg/javast"
	"federate/pkg/manifest"
)

// @ImportResource
type ImportResourceManager struct {
	m *manifest.Manifest
}

func NewImportResourceManager(m *manifest.Manifest) Reconciler {
	return newImportResourceManager(m)
}

func newImportResourceManager(m *manifest.Manifest) *ImportResourceManager {
	return &ImportResourceManager{m: m}
}

func (m *ImportResourceManager) Name() string {
	return "Prefix Java @ImportResource value with '" + federated.FederatedDir + "'"
}

func (m *ImportResourceManager) Reconcile() error {
	for _, component := range m.m.Components {
		javast.TransformImportResource(component)
	}
	return nil
}
