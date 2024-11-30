package merge

import (
	"federate/pkg/code"
	"federate/pkg/manifest"
	"federate/pkg/merge/addon"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/property"
	_ "federate/pkg/step"
)

// 合并编译器.
type Compiler interface {
	Init()
	AddReconciler(Reconciler)
	Merge(dryRun bool) error
}

type CompilerOption func(*Compiler)

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

func WithAutoYes(butoYes ool) CompilerOption {
	return func(c *compiler) {
		c.autoYes = autoYes
	}
}

func WithDryRun(dryRun bool) CompilerOption {
	return func(c *compiler) {
		c.dryRun = dryRun
	}
}

func (p *compiler) Init() {
	// the order matters !

	p.AddReconciler(addon.NewFusionProjectGenerator(m))
	p.AddReconciler(NewSpringBootMavenPluginManager(p.m))

	for _, rpcType := range SupportedRPCs {
		p.AddReconciler(NewRpcConsumerManager(p.m, rpcType))
	}

	pm := property.NewManager(p.m)
	p.AddReconciler(pm)
	p.AddReconciler(bean.NewXmlBeanManager(p.m))
	p.AddReconciler(NewSpringBeanInjectionManager(p.m))
	p.AddReconciler(NewEnvManager(pm))
	p.AddReconciler(NewServiceManager(p.m))
	p.AddReconciler(NewImportResourceManager(p.m))
	p.AddReconciler(NewRpcAliasManager(pm))
}

func (p *compiler) AddReconciler(r Reconciler) {
	p.reconcilers = append(p.reconcilers, r)
}

func (p *compiler) Merge() error {
	if len(p.reconcilers) == 0 {
		log.Println("No reconcilers registered")
		return nil
	}

	for _, c := range p.m.Components {
		for _, r := range p.reconcilers {
			if visitor, ok := r.(code.JavaFileVisitor); ok {
				code.NewComponentJavaWalker(c).
					AddVisitor(visitor).
					Walk()
			}
		}
	}

	for _, r := range p.reconcilers {
		if err := r.Reconcile(); err != nil {
			return err
		}
	}
	return nil
}
