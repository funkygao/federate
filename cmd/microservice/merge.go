package microservice

import (
	"io"
	"log"
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/javast"
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"federate/pkg/merge/addon"
	"federate/pkg/merge/bean"
	"federate/pkg/merge/ledger"
	"federate/pkg/merge/property"
	"federate/pkg/step"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

const (
	defaultCellMaxWidth = 40
	targetSpringXml     = "spring.xml"
)

var (
	yamlConflictCellMaxWidth int
	dryRunMerge              bool
	autoYes                  bool
	silentMode               bool
	noColor                  bool
)

var mergeCmd = &cobra.Command{
	Use:   "consolidate",
	Short: "Merge components into target system following directives of the manifest",
	Long: `The merge command merges components into target system following directives of the manifest.

  See: https://mwhittaker.github.io/publications/service_weaver_HotOS2023.pdf

Example usage:
  federate microservice consolidate -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		doMerge(manifest.Load())
	},
}

func doMerge(m *manifest.Manifest) {
	if silentMode {
		log.SetOutput(io.Discard)
	}
	if noColor {
		color.NoColor = true
	}

	var rpcConsumerManagers []*merge.RpcConsumerManager
	for _, rpc := range merge.SupportedRPCs {
		rpcConsumerManagers = append(rpcConsumerManagers, merge.NewRpcConsumerManager(m, rpc))
	}

	fusionProjectGenerator := addon.NewFusionProjectGenerator(m)
	springBootMavenPlugiManager := merge.NewSpringBootMavenPluginManager(m)
	propertyManager := property.NewManager(m)
	importResourceManager := merge.NewImportResourceManager(m)
	xmlBeanManager := bean.NewXmlBeanManager(m)
	resourceManager := merge.NewResourceManager(m)
	envManager := merge.NewEnvManager(propertyManager)
	injectionManager := merge.NewSpringBeanInjectionManager(m)
	serviceManager := merge.NewServiceManager(m)
	rpcAliasManager := merge.NewRpcAliasManager(propertyManager)

	// TODO v3.0
	compiler := merge.NewCompiler(m, merge.WithDryRun(dryRunMerge), merge.WithAutoYes(autoYes))
	compiler.Merge()

	steps := []step.Step{
		{
			Name: "Generating federated system scaffold",
			Fn: func() {
				if err := fusionProjectGenerator.Reconcile(); err != nil {
					log.Fatalf("%s %v", fusionProjectGenerator.Name(), err)
				}
			}},
		{
			Name: "Instrumentation of spring-boot-maven-plugin",
			Fn: func() {
				if err := springBootMavenPlugiManager.Reconcile(); err != nil {
					log.Fatalf("%s %v", springBootMavenPlugiManager.Name(), err)
				}
			}},
		{
			Name: "Mergeing RPC Consumer XML to reduce redundant resource consumption",
			Fn: func() {
				for _, manager := range rpcConsumerManagers {
					if err := manager.Reconcile(); err != nil {
						log.Fatalf("Error merging %s consumer xml: %v", manager.RPC(), err)
					}
				}
			}},
		{
			Name: "Reconciling Property Conflicts References by Rewriting @Value/@ConfigurationProperties/@RequestMapping",
			Fn: func() {
				if err := propertyManager.Reconcile(); err != nil {
					log.Fatalf("%s %v", propertyManager.Name(), err)
				}
			}},
		{
			Name: "Federated-Copying Resources",
			Fn: func() {
				if err := resourceManager.RecursiveFederatedCopyResources(); err != nil {
					log.Fatalf("Error copying resources: %v", err)
				}
			}},
		{
			Name: "Flat-Copying Resources: reconcile.resources.copy",
			Fn: func() {
				if err := resourceManager.RecursiveFlatCopyResources(); err != nil {
					log.Fatalf("Error merging reconcile.flatCopyResources: %v", err)
				}
			}},
		{
			Name: "Reconciling Spring XML BeanDefinition conflicts by Rewriting XML ref/value-ref/bean/properties-ref",
			Fn: func() {
				xmlBeanManager.Reconcile()
				plan := xmlBeanManager.ReconcilePlan()
				log.Printf("Found bean id conflicts: %d", plan.ConflictCount())
			}},
		{
			Name: "Reconciling Spring Bean Injection conflicts by Rewriting @Resource",
			Fn: func() {
				if err := injectionManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)
				}

				if injectionManager.AutowiredN > 0 {
					log.Printf("Source Code Rewritten, +@Autowired: %d, +@Qualifier: %d", injectionManager.AutowiredN, injectionManager.QualifierN)
				}
			}},
		{
			Name: "Generating Federated Spring Bootstrap XML",
			Fn: func() {
				targetDir := federated.GeneratedResourceBaseDir(m.Main.Name)
				if err := os.MkdirAll(targetDir, 0755); err != nil {
					log.Fatalf("Error creating directory: %v", err)
				}

				targetFile := filepath.Join(targetDir, targetSpringXml)
				fs.GenerateFileFromTmpl("templates/spring.xml", targetFile, m)

				color.Green("🍺 Generated %s", targetFile)
			}},
		{
			Name: "Reconciling ENV variables conflicts",
			Fn: func() {
				if err := envManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Transforming Java @Service/@Component value",
			Fn: func() {
				if err := serviceManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)

				}
			}},
		{
			Name: "Transforming Java @ImportResource value",
			Fn: func() {
				if err := importResourceManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Detecting RPC Provider alias/group conflicts by Rewriting XML",
			Fn: func() {
				rpcAliasManager.Reconcile()
			}},
		{
			Name: "Transforming @Transactional/TransactionTemplate to support multiple PlatformTransactionManager",
			Fn: func() {
				for _, c := range m.Components {
					if err := javast.InjectTransactionManager(c); err != nil {
						log.Fatalf("%v", err)
					}
				}
			}},
		{
			Name: "Post-Instrumentation: display conflict summary guiding you fix fusion-starter",
			Fn: func() {
				ledger.Get().ShowSummary()
			}},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)
}

func init() {
	manifest.RequiredManifestFileFlag(mergeCmd)
	mergeCmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	mergeCmd.Flags().BoolVarP(&silentMode, "silent", "s", false, "Silent or quiet mode")
	mergeCmd.Flags().IntVarP(&yamlConflictCellMaxWidth, "yaml-conflict-cell-width", "w", defaultCellMaxWidth, "Yml files conflict table cell width")
	mergeCmd.Flags().BoolVarP(&dryRunMerge, "dry-run", "d", false, "Perform a dry run without making any changes")
	mergeCmd.Flags().BoolVarP(&noColor, "no-color", "n", false, "Disable colorized output")
	mergeCmd.Flags().BoolVarP(&merge.FailFast, "fail-fast", "f", false, "Fail fast")
}
