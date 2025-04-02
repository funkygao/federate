package insight

import (
	"federate/pkg/javast/ast"
	"federate/pkg/step"
	"github.com/spf13/cobra"
)

var autoYes bool

var wizardCmd = &cobra.Command{
	Use:   "wizard",
	Short: "Step by step insight extraction",
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		runWizard(root)
	},
}

func runWizard(root string) {
	minCount = 6
	ast.Verbosity = 5
	ast.TopK = 30

	steps := []step.Step{
		{
			Name: "Display code building blocks taxonomy",
			Fn: func(bar step.Bar) {
				analyzeTaxonomy(root)
			},
		},
		{
			Name: "Analyze Java extension points",
			Fn: func(bar step.Bar) {
				analyzeExtensions(root)
			},
		},
		{
			Name: "Analyze Git history",
			Fn: func(bar step.Bar) {
				analyzeGitHistory(root)
			},
		},
		{
			Name: "Mining Java AST Graph Knowledge",
			Fn: func(bar step.Bar) {
				runASTCommand(root)
			},
		},
		{
			Name: "Analyze MyBatis MySQL mapper XML files",
			Fn: func(bar step.Bar) {
				analyzeMybatisMapperXML(root)
			},
		},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)
}

func init() {
	wizardCmd.Flags().BoolVarP(&autoYes, "yes", "y", true, "Automatically answer yes to all prompts")
}
