package debug

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"federate/pkg/merge/property"
	"federate/pkg/spring"
	"federate/pkg/step"
	"github.com/spf13/cobra"
)

var autoYes bool

var wizardCmd = &cobra.Command{
	Use:   "wizard",
	Short: "Step by step guide for detecting potential conflicts",
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		runWizard(m)
	},
}

func runWizard(m *manifest.Manifest) {
	propertyManager := property.NewManager(m)
	rpcAliasManager := merge.NewRpcAliasManager(propertyManager)

	steps := []step.Step{
		{
			Name: "Detect spring.xml multiple ref to the same bean",
			Fn: func() {
				listRef(m, spring.New(false))
			},
		},
		{
			Name: "Detect one RPC Interface with multiple alias/group",
			Fn: func() {
				propertyManager.Silent().Prepare()
				merge.RunReconcile(rpcAliasManager, nil)
			},
		},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)
}

func init() {
	manifest.RequiredManifestFileFlag(wizardCmd)
	wizardCmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
}
