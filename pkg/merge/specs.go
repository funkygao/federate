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

// 有准备阶段的合并编译调和器.
type PreparableReconciler interface {
	Reconciler

	Prepare() error
}

// 可插拔的合并编译调和器.
type PluginReconciler interface {
	Reconciler

	// 插件初始化
	Init(*manifest.Manifest) error
}

// 只检测冲突，不调和
type DetectOnlyReconciler interface {
	Reconciler

	DetectOnly() bool
}

type XMLGenerator interface {
	GeneratedXML() []string
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
