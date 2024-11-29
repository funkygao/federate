package merge

import (
	"federate/pkg/code"
	"federate/pkg/manifest"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/property"
)

// 合并编译器.
type Compiler interface {
	Prepare()

	AddReconciler(Reconciler)

	// 合并编译.
	Compile(dryRun bool) error
}

func NewCompiler(m *manifest.Manifest) Compiler {
	return &compiler{
		m:           m,
		reconcilers: make([]Reconciler, 0),
	}
}

type compiler struct {
	m           *manifest.Manifest
	reconcilers []Reconciler
}

func (p *compiler) Prepare() {
	// the order matters !
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

func (p *compiler) Compile(dryRun bool) error {
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
		if err := r.Reconcile(dryRun); err != nil {
			return err
		}
	}
	return nil
}
