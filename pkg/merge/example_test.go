package merge

import (
	"testing"

	"federate/pkg/manifest"
)

func ExamplePackager(t *testing.T) {
	m := manifest.Load()

	packager := NewMergePackager(m)

	// 添加一个模拟的 Reconciler
	packager.AddReconciler(&mockReconciler{})

	if err := packager.Execute(true); err != nil {
		t.Fatalf("Error executing packager: %v", err)
	}

	t.Log("Merge packager executed successfully")
}

type mockReconciler struct{}

func (m *mockReconciler) Name() string {
	return "mock"
}

func (m *mockReconciler) Reconcile(dryRun bool) error {
	return nil
}
