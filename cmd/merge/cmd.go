package merge

import (
	"io"
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"federate/pkg/step"
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
)

var MergeCmd = &cobra.Command{
	Use:   "consolidate",
	Short: "Merge components into target system following directives of the manifest",
	Long: `The merge command merges components into target system following directives of the manifest.

  See: https://mwhittaker.github.io/publications/service_weaver_HotOS2023.pdf

Example usage:
  federate microservice consolidate -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.LoadManifest()
		doMerge(m)
	},
}

func doMerge(m *manifest.Manifest) {
	if silentMode {
		log.SetOutput(io.Discard)
	}

	rpcTypes := []string{merge.RpcJsf, merge.RpcDubbo}
	var rpcConsumerManagers []*merge.RpcConsumerManager
	for _, rpc := range rpcTypes {
		rpcConsumerManagers = append(rpcConsumerManagers, merge.NewRpcConsumerManager(rpc))
	}

	propertySourcesManager := merge.NewPropertySourcesManager(m)
	xmlBeanManager := merge.NewXmlBeanManager(m)
	resourceManager := merge.NewResourceManager()
	injectionManager := merge.NewSpringBeanInjectionManager()

	steps := []step.Step{
		{
			Name: "Generating target system scaffold",
			Fn: func() {
				scaffoldTargetSystem(m)
			}},
		{
			Name: "Instrumentation of spring-boot-maven-plugin",
			Fn: func() {
				instrumentPomForFederatePackaging(m) // 代码插桩
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
			Name: "Flat-Copying Resources: reconcile.resources.flatCopy",
			Fn: func() {
				recursiveFlatCopyResources(m, resourceManager)
			}},
		{
			Name: "Prepare Merging PropertySources of .properties files",
			Fn: func() {
				prepareMergePropertiesFiles(m, propertySourcesManager)
			}},
		{
			Name: "Prepare Merging PropertySources of application.yml, handling 'spring.profiles.include'",
			Fn: func() {
				prepareMergeApplicationYaml(m, propertySourcesManager)
			}},
		{
			Name: "Reconciling Placeholder conflicts references by rewriting .java/.xml files",
			Fn: func() {
				reconcilePropertiesConflicts(m, propertySourcesManager)
			}},
		{
			Name: "Reconciling Spring XML BeanDefinition conflicts",
			Fn: func() {
				reconcileTargetXmlBeanConflicts(m, xmlBeanManager)
			}},
		{
			Name: "Reconciling Spring Bean Injection conflicts",
			Fn: func() {
				reconcileBeanInjectionConflicts(m, injectionManager)
			}},
		{
			Name: "Generating Federated Spring Bootstrap XML",
			Fn: func() {
				generateSpringBootstrapXML(m)
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
}
