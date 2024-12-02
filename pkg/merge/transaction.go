package merge

import (
	"federate/pkg/javast"
	"federate/pkg/manifest"
)

// 处理 @Transactional
type TransactionManager struct {
	m *manifest.Manifest
}

func NewTransactionManager(m *manifest.Manifest) Reconciler {
	return &TransactionManager{m: m}
}

func (m *TransactionManager) Name() string {
	return "Transforming @Transactional/TransactionTemplate to support multiple PlatformTransactionManager"
}

func (m *TransactionManager) Reconcile() error {
	for _, c := range m.m.Components {
		if err := javast.InjectTransactionManager(c); err != nil {
			return err
		}
	}
	return nil
}
