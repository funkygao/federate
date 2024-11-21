package merge

import (
	"federate/pkg/manifest"
)

type fusionStarterManager struct {
	m *manifest.Manifest
}

func NewFusionStarterManager(m *manifest.Manifest) Reconciler {
	return &fusionStarterManager{m: m}
}

func (fs *fusionStarterManager) Reconcile(dryRun bool) error {
	return nil
}
