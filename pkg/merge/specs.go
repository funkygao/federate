package merge

// 合并编译调和器.
type Reconciler interface {
	Name() string

	// 执行调和动作.
	Reconcile() error
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
