package merge

import (
	"testing"

	"federate/pkg/manifest"
)

func ExampleCompiler(t *testing.T) {
	m := manifest.Load()

	compiler := NewCompiler(m)
	compiler.Init()

	// 添加一个模拟的 Reconciler
	compiler.AddReconciler(&mockReconciler{})

	if err := compiler.Merge(); err != nil {
		t.Fatalf("Error executing compiler: %v", err)
	}

	t.Log("Merge compiler executed successfully")
}

type mockReconciler struct{}

func (m *mockReconciler) Name() string {
	return "mock"
}

func (m *mockReconciler) Reconcile() error {
	return nil
}
