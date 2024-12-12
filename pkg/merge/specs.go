package merge

import (
	"fmt"

	"federate/pkg/manifest"
	"federate/pkg/step"
)

// 合并编译调和器.
type Reconciler interface {
	Name() string
}

// 不带进度条的 Reconciler
type SimpleReconciler interface {
	Reconciler

	Reconcile() error
}

// 带进度条的 Reconciler
type ProgressReconciler interface {
	Reconciler

	Reconcile(step.Bar) error
}

func RunReconcile(r Reconciler, bar step.Bar) error {
	switch rec := r.(type) {
	case SimpleReconciler:
		return rec.Reconcile()
	case ProgressReconciler:
		return rec.Reconcile(bar)
	default:
		return fmt.Errorf("reconciler %s does not implement any Reconcile method", r.Name())
	}
}

// 有准备阶段的合并编译调和器.
type Preparer interface {
	Reconciler

	// 准备工作
	Prepare() error
}

// 可以扩展的合并编译调和器插件.
type PluginReconciler interface {
	Reconciler

	// 插件初始化
	Init(m *manifest.Manifest) error
}
