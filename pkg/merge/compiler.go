package merge

import (
	"fmt"
	"log"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/ledger"
	"federate/pkg/merge/property"
	"federate/pkg/step"
	"github.com/fatih/color"
	"github.com/schollz/progressbar/v3"
)

// 合并编译器.
type Compiler interface {
	// 初始化，注册内置调和器
	Init() Compiler

	WithOption(CompilerOption) Compiler

	// 注册调和器
	AddReconciler(...Reconciler) Compiler

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
	if len(p.m.Main.Reconcile.Transformers) > 0 {
		p.orchestrateReconcilers()
	} else {
		p.loadDefaultReconcilers()
	}

	// Load plugin reconcilers
	if pluginReconcilers, err := LoadPluginReconcilers(federated.FederatePluginsDir, p.m); err != nil {
		log.Printf("Error loading plugin reconcilers: %v", err)
	} else {
		p.AddReconciler(pluginReconcilers...)
	}

	return p
}

func (p *compiler) loadDefaultReconcilers() {
	// the order matters !
	p.AddReconciler(NewFusionProjectGenerator(p.m))
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
	p.AddReconciler(NewServiceManager(p.m))
	p.AddReconciler(NewImportResourceManager(p.m))
	p.AddReconciler(NewTransactionManager(p.m))
	p.AddReconciler(NewEnvManager(pm))
	p.AddReconciler(NewRpcAliasManager(pm))
	p.AddReconciler(NewJavaAstTransformer(p.m))
}

func (p *compiler) orchestrateReconcilers() {
	for _, tf := range p.m.Main.Reconcile.Transformers {
		if rc := p.createReconciler(tf); rc != nil {
			p.AddReconciler(rc)
		}
	}

	log.Fatalf("Not implemented")
}

func (p *compiler) createReconciler(tf manifest.TransformerSpec) Reconciler {
	return nil
}

func (p *compiler) AddReconciler(reconcilers ...Reconciler) Compiler {
	p.reconcilers = append(p.reconcilers, reconcilers...)
	return p
}

func (p *compiler) Merge() error {
	if len(p.reconcilers) == 0 {
		return fmt.Errorf("No reconcilers registered: call Init() before Merge()")
	}

	if p.dryRun {
		p.displayDAG()
		return nil
	}

	var steps = []step.Step{
		step.Step{
			Name: "Consolidation Plan",
			FnWithProgress: func(bar *progressbar.ProgressBar) {
				p.displayDAG()
				bar.Add(100)
			}},
	}

	// 先执行 Preparer
	steps = p.prepareReconcilers(steps)

	for _, r := range p.reconcilers {
		steps = append(steps, step.Step{
			Name: r.Name(),
			FnWithProgress: func(bar *progressbar.ProgressBar) {
				if err := RunReconcile(r, bar); err != nil {
					log.Fatalf("[%s] %v", r.Name(), err)
				}
			},
		})
	}

	// last step: report
	steps = append(steps, step.Step{
		Name: "Consolidation Summary Dump to report.json",
		FnWithProgress: func(bar *progressbar.ProgressBar) {
			ledger.Get().ShowSummary()
			bar.Add(50)
			ledger.Get().SaveToFile("report.json")
			bar.Finish()
		}})

	step.AutoConfirm = p.autoYes
	step.Run(steps)
	return nil
}

func (p *compiler) prepareReconcilers(steps []step.Step) []step.Step {
	// prepare if nec: plugin might also implement Preparer
	for _, r := range p.reconcilers {
		if p, ok := r.(Preparer); ok {
			steps = append(steps, step.Step{
				Name: color.CyanString(p.Name() + " *"),
				FnWithProgress: func(bar *progressbar.ProgressBar) {
					p.Prepare()
				},
			})
		}
	}

	return steps
}

func (p *compiler) displayDAG() {
	log.Printf("Reconcilers [ Legend: %s Preparer | %s Plugin ]", color.CyanString("■"), color.GreenString("■"))

	for _, r := range p.reconcilers {
		indicator := "  "
		if _, ok := r.(Preparer); ok {
			indicator = color.CyanString("■ ")
		}
		if _, ok := r.(PluginReconciler); ok {
			indicator = color.GreenString("■ ")
		}

		log.Printf("  %s%s%s", "├── ", indicator, r.Name())
	}
	// summary is step instead of reconciler
	log.Printf("  └──   Consolidation Summary Dump to report.json")
}
