package merge

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
	"federate/pkg/merge/bean"
	"federate/pkg/merge/property"
	"federate/pkg/merge/transformer"
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

	rpcTypes := []string{merge.RpcJsf, merge.RpcDubbo}
	var rpcConsumerManagers []*merge.RpcConsumerManager
	for _, rpc := range rpcTypes {
		rpcConsumerManagers = append(rpcConsumerManagers, merge.NewRpcConsumerManager(m, rpc))
	}

	propertyManager := property.NewManager(m)
	xmlBeanManager := bean.NewXmlBeanManager(m)
	resourceManager := merge.NewResourceManager(m)
	envManager := merge.NewEnvManager(m, propertyManager)
	injectionManager := merge.NewSpringBeanInjectionManager(m)
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
			Name: "Mergeing RPC Consumer XML to reduce redundant resource consumption",
			Fn: func() {
				mergeRpcConsumerXml(rpcConsumerManagers)
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
			Name: "Analyze All Property and Identify Conflicts",
			Fn: func() {
				identifyPropertyConflicts(propertyManager)
			}},
		{
			Name: "Reconciling Property Conflicts References by Rewriting @Value/@ConfigurationProperties/@RequestMapping",
			Fn: func() {
				reconcilePropertiesConflicts(propertyManager)
			}},
		{
			Name: "Reconciling Spring XML BeanDefinition conflicts by Rewriting XML ref/value-ref/bean/properties-ref",
			Fn: func() {
				xmlBeanManager.Reconcile(dryRunMerge)
				plan := xmlBeanManager.ReconcilePlan()
				log.Printf("Found bean id conflicts: %d", plan.ConflictCount())
			}},
		{
			Name: "Reconciling Spring Bean Injection conflicts by Rewriting @Resource",
			Fn: func() {
				if err := injectionManager.Reconcile(dryRunMerge); err != nil {
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
				if err := envManager.Reconcile(dryRunMerge); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Transforming Java @Service/@Component value",
			Fn: func() {
				if err := serviceManager.Reconcile(dryRunMerge); err != nil {
					log.Fatalf("%v", err)

				}
			}},
		{
			Name: "Transforming Java @ImportResource value",
			Fn: func() {
				importResourceManager := merge.NewImportResourceManager(m)
				if err := importResourceManager.Reconcile(dryRunMerge); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Detecting RPC Provider alias/group conflicts by Rewriting XML",
			Fn: func() {
				rpcAliasManager.Reconcile(dryRunMerge)
			}},
		{
			Name: "Transforming @Transactional to support multiple PlatformTransactionManager",
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
				transformer.Get().ShowSummary()
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
	MergeCmd.Flags().BoolVarP(&merge.FailFast, "fail-fast", "f", false, "Fail fast")
}
