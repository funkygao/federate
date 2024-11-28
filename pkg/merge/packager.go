package merge

import (
	"federate/pkg/code"
	"federate/pkg/manifest"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/property"
)

// 合并打包器.
type Packager interface {
	AddReconciler(Reconciler)
	Execute(dryRun bool) error
}

// 准备合并打包器，并自动注册内置 Reconciler.
func NewMergePackager(m *manifest.Manifest) Packager {
	p := &packager{
		m:           m,
		reconcilers: make([]Reconciler, 0),
	}

	pm := property.NewManager(m)

	// the order matters !
	for _, rpcType := range SupportedRPCs {
		p.AddReconciler(NewRpcConsumerManager(m, rpcType))
	}
	p.AddReconciler(pm)
	p.AddReconciler(bean.NewXmlBeanManager(m))
	p.AddReconciler(NewSpringBeanInjectionManager(m))
	p.AddReconciler(NewEnvManager(m, pm))
	p.AddReconciler(NewServiceManager(m))
	p.AddReconciler(NewImportResourceManager(m))
	p.AddReconciler(NewRpcAliasManager(pm))

	return p
}

type packager struct {
	m           *manifest.Manifest
	reconcilers []Reconciler
}

func (p *packager) AddReconciler(r Reconciler) {
	p.reconcilers = append(p.reconcilers, r)
}

func (p *packager) Execute(dryRun bool) error {
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
