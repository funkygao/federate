package merge

import (
	"federate/pkg/manifest"
)

type trxManager struct {
	m *manifest.Manifest
}

func NewTrxManager(m *manifest.Manifest) Reconciler {
	return &trxManager{m: m}
}

func (t *trxManager) Reconcile(dryRun bool) error {
	return nil
}
