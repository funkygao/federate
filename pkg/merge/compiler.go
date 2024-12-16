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
)

// 合并编译器.
type Compiler interface {
	// 加载调和器
	Init()

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
	silent  bool
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

func WithSilent(silent bool) CompilerOption {
	return func(c Compiler) {
		if cc, ok := c.(*compiler); ok {
			cc.silent = silent
		}
	}
}

func (p *compiler) WithOption(opt CompilerOption) Compiler {
	opt(p)
	return p
}

func (p *compiler) Init() {
	if len(p.m.Main.Reconcile.Transformers) > 0 {
		p.orchestrateReconcilers()
	} else {
		p.loadDefaultReconcilers()
	}

	p.loadPluginReconcilers()
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

func (p *compiler) loadPluginReconcilers() {
	if pluginReconcilers, err := LoadPluginReconcilers(federated.FederatePluginsDir, p.m); err != nil {
		log.Printf("Error loading plugin reconcilers: %v", err)
	} else {
		p.AddReconciler(pluginReconcilers...)
	}
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
		p.showOverview()
		return nil
	}

	// 先执行 PreparableReconciler
	steps := p.prepareReconcilers(nil)

	// 再编排内置 Reconciler
	for _, r := range p.reconcilers {
		if _, skipOnSilent := r.(DetectOnlyReconciler); skipOnSilent && p.silent {
			continue
		}

		steps = append(steps, step.Step{
			Name: r.Name(),
			Fn: func(bar step.Bar) {
				if err := RunReconcile(r, bar); err != nil {
					log.Fatalf("[%s] %v", r.Name(), err)
				}
			},
		})
	}

	step.AutoConfirm = p.autoYes
	step.Run(steps)

	if !p.silent {
		ledger.Get().ShowSummary()
	}
	ledger.Get().SaveToFile("report.json")

	return nil
}

func (p *compiler) prepareReconcilers(steps []step.Step) []step.Step {
	if steps == nil {
		steps = make([]step.Step, 0, 20)
	}
	for _, r := range p.reconcilers {
		if p, ok := r.(PreparableReconciler); ok {
			steps = append(steps, step.Step{
				Name: color.CyanString(p.Name() + " *"),
				Fn: func(bar step.Bar) {
					p.Prepare()
				},
			})
		}
	}

	return steps
}

func (p *compiler) showOverview() {
	log.Printf("Reconcilers [ Legend: %s Preparable | %s Plugin ]", color.CyanString("■"), color.GreenString("■"))

	for i, r := range p.reconcilers {
		prefix := "├── "
		if i == len(p.reconcilers)-1 {
			prefix = "└── "
		}

		indicator := "  "
		if _, ok := r.(PreparableReconciler); ok {
			indicator = color.CyanString("■ ")
		}
		if _, ok := r.(PluginReconciler); ok {
			indicator = color.GreenString("■ ")
		}

		log.Printf("  %s%s%s", prefix, indicator, r.Name())
	}
}
