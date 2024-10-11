package merge

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"

	"federate/pkg/manifest"
	"federate/pkg/merge"
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
)

var MergeCmd = &cobra.Command{
	Use:   "merge",
	Short: "Merge resources of components from the manifest file",
	Long: `The merge command generates merged resource files based on the provided manifest file. 

  See: https://mwhittaker.github.io/publications/service_weaver_HotOS2023.pdf

Example usage:
  federate microservice merge -i manifest.yaml`,
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

	propertySourcesManager := merge.NewPropertySourcesManager()
	xmlBeanManager := merge.NewXmlBeanManager(m)
	resourceManager := merge.NewResourceManager()
	injectionManager := merge.NewSpringBeanInjectionManager()

	steps := []struct {
		name string
		fn   func()
	}{
		{"Generating federate system scaffold", func() {
			scaffoldFederatedSystem(m)
		}},
		{"Reconciling ENV variables conflicts", func() {
			reconcileEnvConflicts(m)
		}},
		{"Mergeing RPC Consumer XML", func() {
			mergeRpcConsumerXml(m, rpcConsumerManagers)
		}},
		{"Recursively Federated-Copying Resources to federated resources dir", func() {
			recursiveFederatedCopyResources(m, resourceManager)
		}},
		{"Recursively Flat-Copying Resources: reconcile.flatCopyResources", func() {
			recursiveFlatCopyResources(m, resourceManager)
		}},
		{"Prepare Merging PropertySources of .properties files", func() {
			prepareMergePropertiesFiles(m, propertySourcesManager)
		}},
		{"Prepare Merging PropertySources of application.yml, handling 'spring.profiles.include'", func() {
			prepareMergeApplicationYaml(m, propertySourcesManager)
		}},
		{"Reconciling placeholder conflicts references by updating .java/.xml files", func() {
			reconcilePropertiesConflicts(m, propertySourcesManager)
		}},
		{"Reconciling Target XML beans conflicts", func() {
			reconcileTargetXmlBeanConflicts(m, xmlBeanManager)
		}},
		{"Reconciling Spring Bean injection conflicts", func() {
			reconcileBeanInjectionConflicts(m, injectionManager)
		}},
		{"Generating Spring Bootstrap XML", func() {
			generateSpringBootstrapXML(m)
		}},
	}

	totalSteps := len(steps)
	for i, step := range steps {
		promptToProceed(i+1, totalSteps, step.name)
		step.fn()
	}
}

func promptToProceed(seq, total int, step string) {
	c := color.New(color.FgMagenta)
	c.Printf("Step [%d/%d] %s ...", seq, total, step)
	if autoYes {
		fmt.Println()
		return
	}
	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')
}

func init() {
	manifest.RequiredManifestFileFlag(MergeCmd)
	MergeCmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	MergeCmd.Flags().BoolVarP(&silentMode, "silent", "s", false, "Silent or quiet mode")
	MergeCmd.Flags().IntVarP(&yamlConflictCellMaxWidth, "yaml-conflict-cell-width", "w", defaultCellMaxWidth, "Yml files conflict table cell width")
	MergeCmd.Flags().BoolVarP(&dryRunMerge, "dry-run", "d", false, "Perform a dry run without making any changes")
}
