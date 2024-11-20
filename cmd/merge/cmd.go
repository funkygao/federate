package merge

import (
	"io"
	"log"
	"os"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/merge"
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

var MergeCmd = &cobra.Command{
	Use:   "consolidate",
	Short: "Merge components into target system following directives of the manifest",
	Long: `The merge command merges components into target system following directives of the manifest.

  See: https://mwhittaker.github.io/publications/service_weaver_HotOS2023.pdf

Example usage:
  federate microservice consolidate -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doMerge(m)
	},
}

func doMerge(m *manifest.Manifest) {
	if silentMode {
		log.SetOutput(io.Discard)
	}
	if noColor {
		color.NoColor = true
	}

	rpcTypes := []string{merge.RpcJsf, merge.RpcDubbo}
	var rpcConsumerManagers []*merge.RpcConsumerManager
	for _, rpc := range rpcTypes {
		rpcConsumerManagers = append(rpcConsumerManagers, merge.NewRpcConsumerManager(rpc))
	}

	propertyManager := merge.NewPropertyManager(m)
	xmlBeanManager := merge.NewXmlBeanManager(m)
	resourceManager := merge.NewResourceManager()
	injectionManager := merge.NewSpringBeanInjectionManager()
	serviceManager := merge.NewServiceManager(m)
	rpcAliasManager := merge.NewRpcAliasManager(propertyManager)

	steps := []step.Step{
		{
			Name: "Generating federated system scaffold",
			Fn: func() {
				scaffoldTargetSystem(m)
			}},
		{
			Name: "Instrumentation of spring-boot-maven-plugin",
			Fn: func() {
				InstrumentPomForFederatePackaging(m) // 代码插桩
			}},
		{
			Name: "Reconciling ENV variables conflicts",
			Fn: func() {
				merge.ReconcileEnvConflicts(m)
				color.Green("🍺 ENV variables conflicts reconciled")
			}},
		{
			Name: "Mergeing RPC Consumer XML to reduce redundant resource consumption",
			Fn: func() {
				mergeRpcConsumerXml(m, rpcConsumerManagers)
			}},
		{
			Name: "Federated-Copying Resources",
			Fn: func() {
				if err := resourceManager.RecursiveFederatedCopyResources(m); err != nil {
					log.Fatalf("Error copying resources: %v", err)
				}

				color.Green("🍺 Resources recursively federated copied")
			}},
		{
			Name: "Flat-Copying Resources: reconcile.resources.copy",
			Fn: func() {
				if err := resourceManager.RecursiveFlatCopyResources(m); err != nil {
					log.Fatalf("Error merging reconcile.flatCopyResources: %v", err)
				}
				color.Green("🍺 Resources recursively flat copied")
			}},
		{
			Name: "Analyze All Property and Identify Conflicts",
			Fn: func() {
				identifyPropertyConflicts(m, propertyManager)
			}},
		{
			Name: "Reconciling Property Conflicts References by Rewriting @Value/@ConfigurationProperties/@RequestMapping",
			Fn: func() {
				reconcilePropertiesConflicts(m, propertyManager)
			}},
		{
			Name: "Reconciling Spring XML BeanDefinition conflicts by Rewriting XML ref/value-ref/bean/properties-ref",
			Fn: func() {
				xmlBeanManager.ReconcileTargetConflicts(dryRunMerge)
				plan := xmlBeanManager.ReconcilePlan()
				log.Printf("Found bean id conflicts: %d", plan.ConflictCount())
				color.Green("🍺 Spring XML Beans conflicts reconciled")
			}},
		{
			Name: "Reconciling Spring Bean Injection conflicts by Rewriting @Resource",
			Fn: func() {
				if err := injectionManager.Reconcile(m, dryRunMerge); err != nil {
					log.Fatalf("%v", err)
				}

				if injectionManager.AutowiredN > 0 {
					log.Printf("Source Code Rewritten, +@Autowired: %d, +@Qualifier: %d", injectionManager.AutowiredN, injectionManager.QualifierN)
				}
				color.Green("🍺 Java code Spring Bean Injection conflicts reconciled")
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
			Name: "Transforming Java @Service value",
			Fn: func() {
				if err := serviceManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)

				}
				color.Green("🍺 Java @Service and corresponding spring.xml ref transformed")
			}},
		{
			Name: "Transforming Java @ImportResource value",
			Fn: func() {
				importResourceManager := merge.NewImportResourceManager(m)
				if err := importResourceManager.Reconcile(); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Detecting RPC Provider alias/group conflicts by Rewriting XML",
			Fn: func() {
				rpcAliasManager.Reconcile()
			}},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)
}

func init() {
	manifest.RequiredManifestFileFlag(MergeCmd)
	MergeCmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	MergeCmd.Flags().BoolVarP(&silentMode, "silent", "s", false, "Silent or quiet mode")
	MergeCmd.Flags().IntVarP(&yamlConflictCellMaxWidth, "yaml-conflict-cell-width", "w", defaultCellMaxWidth, "Yml files conflict table cell width")
	MergeCmd.Flags().BoolVarP(&dryRunMerge, "dry-run", "d", false, "Perform a dry run without making any changes")
	MergeCmd.Flags().BoolVarP(&noColor, "no-color", "n", false, "Disable colorized output")
}
