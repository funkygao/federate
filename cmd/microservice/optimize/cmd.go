package optimize

import (
	"federate/pkg/manifest"
	"federate/pkg/step"
	"github.com/spf13/cobra"
)

var (
	autoYes bool
)

var CmdGroup = &cobra.Command{
	Use:   "optimize",
	Short: "Identify potential areas for optimization",
	Long:  `The optimize command identify potential areas for optimization.`,
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.Load()
		optimize(manifest)
	},
}

func optimize(m *manifest.Manifest) {
	steps := []step.Step{
		{
			Name: "Detect similar classes for potential duplicate coding",
			Fn: func() {
				showDuplicates(m)
			}},
		{
			Name: "Optimize project JAR dependencies",
			Fn: func() {
			}},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)

}

func init() {
	CmdGroup.AddCommand(duplicateCmd, dependencyCmd)

	manifest.RequiredManifestFileFlag(CmdGroup)
	CmdGroup.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	CmdGroup.Flags().IntVarP(&optimizeVerbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
}
