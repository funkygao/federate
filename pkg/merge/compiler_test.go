package merge

import (
	"testing"

	"federate/pkg/manifest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockReconciler is a mock implementation of the Reconciler interface
type MockReconciler struct {
	mock.Mock
}

func (m *MockReconciler) Name() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockReconciler) Reconcile() error {
	args := m.Called()
	return args.Error(0)
}

func TestNewCompiler(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m)

	assert.NotNil(t, c, "NewCompiler should return a non-nil Compiler")
	assert.IsType(t, &compiler{}, c, "NewCompiler should return a *compiler")
}

func TestCompiler_Init(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m).(*compiler)

	c.Init()

	assert.Greater(t, len(c.reconcilers), 0, "Init should register some reconcilers")
}

func TestCompiler_AddReconciler(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m).(*compiler)

	mockReconciler := new(MockReconciler)
	c.AddReconciler(mockReconciler)

	assert.Contains(t, c.reconcilers, mockReconciler, "AddReconciler should add the reconciler to the list")
}

func TestCompiler_Merge(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m).(*compiler)

	mockReconciler1 := new(MockReconciler)
	mockReconciler1.On("Name").Return("MockReconciler1")
	mockReconciler1.On("Reconcile").Return(nil)

	mockReconciler2 := new(MockReconciler)
	mockReconciler2.On("Name").Return("MockReconciler2")
	mockReconciler2.On("Reconcile").Return(nil)

	c.AddReconciler(mockReconciler1)
	c.AddReconciler(mockReconciler2)

	err := c.Merge()

	assert.NoError(t, err, "Merge should not return an error")
	mockReconciler1.AssertCalled(t, "Reconcile")
	mockReconciler2.AssertCalled(t, "Reconcile")
}

func TestCompiler_WithOption(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m).(*compiler)

	c.WithOption(WithAutoYes(true))
	assert.True(t, c.autoYes, "WithAutoYes option should set autoYes to true")

	c.WithOption(WithDryRun(true))
	assert.True(t, c.dryRun, "WithDryRun option should set dryRun to true")
}

func TestCompiler_MergeWithNoReconcilers(t *testing.T) {
	m := &manifest.Manifest{}
	c := NewCompiler(m).(*compiler)

	err := c.Merge()

	assert.Error(t, err, "Merge should return an error when no reconcilers are registered")
	assert.Contains(t, err.Error(), "No reconcilers registered", "Error message should mention no reconcilers")
}
