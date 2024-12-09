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
	Short: "Identify potential opportunities for optimization",
	Long:  `The optimize command identify potential opportunities for optimization.`,
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
			Name: "Analyze HITS: Hyperlink-Induced Topic Search",
			Fn: func() {
				analyzeHITS(m)
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
	CmdGroup.AddCommand(duplicateCmd, hitsCmd, dependencyCmd)

	manifest.RequiredManifestFileFlag(CmdGroup)
	CmdGroup.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	CmdGroup.Flags().IntVarP(&optimizeVerbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
}
