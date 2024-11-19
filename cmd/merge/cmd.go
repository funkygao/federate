package merge

import (
	"io"
	"log"

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
				reconcileEnvConflicts(m)
			}},
		{
			Name: "Mergeing RPC Consumer XML to reduce redundant resource consumption",
			Fn: func() {
				mergeRpcConsumerXml(m, rpcConsumerManagers)
			}},
		{
			Name: "Federated-Copying Resources",
			Fn: func() {
				recursiveFederatedCopyResources(m, resourceManager)
			}},
		{
			Name: "Flat-Copying Resources: reconcile.resources.copy",
			Fn: func() {
				recursiveFlatCopyResources(m, resourceManager)
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
				reconcileTargetXmlBeanConflicts(m, xmlBeanManager)
			}},
		{
			Name: "Reconciling Spring Bean Injection conflicts by Rewriting @Resource",
			Fn: func() {
				reconcileBeanInjectionConflicts(m, injectionManager)
			}},
		{
			Name: "Generating Federated Spring Bootstrap XML",
			Fn: func() {
				generateSpringBootstrapXML(m)
			}},
		{
			Name: "Transforming Java @Service value",
			Fn: func() {
				transformServiceValue(serviceManager)
			}},
		{
			Name: "Transforming Java @ImportResource value",
			Fn: func() {
				importResourceManager := merge.NewImportResourceManager()
				if err := importResourceManager.Reconcile(m); err != nil {
					log.Fatalf("%v", err)
				}
			}},
		{
			Name: "Reconciling RPC alias/group naming conflicts by Rewriting XML",
			Fn: func() {
				reconcileRpcAliasConflict(rpcAliasManager)
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
