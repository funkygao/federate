package merge

import (
	"federate/pkg/merge/ledger"
)

type ReconcileSummary struct {
}

func newSummary() Reconciler {
	return &ReconcileSummary{}
}

func (m *ReconcileSummary) Name() string {
	return "Reconcile Summary"
}

func (m *ReconcileSummary) Reconcile() error {
	ledger.Get().ShowSummary()
	return nil
}
