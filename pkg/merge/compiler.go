package merge

import (
	"fmt"
	"log"

	"federate/pkg/code"
	"federate/pkg/manifest"
	"federate/pkg/merge/addon"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/property"
	"federate/pkg/step"
)

// 合并编译器.
type Compiler interface {
	// 初始化，注册内置调和器
	Init() Compiler

	WithOption(CompilerOption) Compiler

	// 注册调和器
	AddReconciler(Reconciler) Compiler

	// 合并编译
	Merge() error
}

type CompilerOption func(Compiler)

func NewCompiler(m *manifest.Manifest, opts ...CompilerOption) Compiler {
	c := &compiler{
		m:           m,
		reconcilers: make([]Reconciler, 0),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

type compiler struct {
	m           *manifest.Manifest
	reconcilers []Reconciler

	dryRun  bool
	autoYes bool
}

func WithAutoYes(autoYes bool) CompilerOption {
	return func(c Compiler) {
		if cc, ok := c.(*compiler); ok {
			cc.autoYes = autoYes
		}
	}
}

func WithDryRun(dryRun bool) CompilerOption {
	return func(c Compiler) {
		if cc, ok := c.(*compiler); ok {
			cc.dryRun = dryRun
		}
	}
}

func (p *compiler) WithOption(opt CompilerOption) Compiler {
	opt(p)
	return p
}

func (p *compiler) Init() Compiler {
	// the order matters !
	p.AddReconciler(addon.NewFusionProjectGenerator(p.m))
	p.AddReconciler(NewSpringBootMavenPluginManager(p.m))

	for _, rpcType := range SupportedRPCs {
		p.AddReconciler(NewRpcConsumerManager(p.m, rpcType))
	}

	// 属性管理器修改 component 目录，需要在 ResourceManager 之前处理后，才能拷贝到目标目录
	pm := property.NewManager(p.m)
	p.AddReconciler(pm)

	p.AddReconciler(NewResourceManager(p.m))
	p.AddReconciler(bean.NewXmlBeanManager(p.m))
	p.AddReconciler(NewSpringBeanInjectionManager(p.m))
	p.AddReconciler(NewSpringXmlMerger(p.m))
	p.AddReconciler(NewEnvManager(pm))
	p.AddReconciler(NewServiceManager(p.m))
	p.AddReconciler(NewImportResourceManager(p.m))
	p.AddReconciler(NewRpcAliasManager(pm))
	p.AddReconciler(NewTransactionManager(p.m))

	// prepare if nec
	for _, r := range p.reconcilers {
		if p, ok := r.(Preparer); ok {
			p.Prepare()
		}
	}

	return p
}

func (p *compiler) AddReconciler(r Reconciler) Compiler {
	p.reconcilers = append(p.reconcilers, r)
	return p
}

func (p *compiler) Merge() error {
	if len(p.reconcilers) == 0 {
		return fmt.Errorf("No reconcilers registered: call Init() before Merge()")
	}

	if false {
		for _, c := range p.m.Components {
			for _, r := range p.reconcilers {
				if visitor, ok := r.(code.JavaFileVisitor); ok {
					code.NewComponentJavaWalker(c).
						AddVisitor(visitor).
						Walk()
				}
			}
		}
	}

	var steps = []step.Step{}
	for _, r := range p.reconcilers {
		steps = append(steps, step.Step{
			Name: r.Name(),
			Fn: func() {
				if err := r.Reconcile(); err != nil {
					log.Fatalf("[%s] %v", r.Name(), err)
				}
			},
		})
	}

	step.AutoConfirm = p.autoYes
	step.Run(steps)
	return nil
}