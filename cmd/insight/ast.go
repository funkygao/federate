package insight

import (
	"log"

	"federate/pkg/javast"
	"federate/pkg/javast/ast"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var astCmd = &cobra.Command{
	Use:   "javaast <dir>",
	Short: "Mining AST graph knowledge from a Java source directory",
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		runASTCommand(root)
	},
}

func runASTCommand(root string) {
	driver := javast.NewJavastDriver()
	astInfo, err := driver.ExtractAST(root)
	if err != nil {
		log.Fatalf("%s %v", root, err)
	}

	astInfo.ShowReport()
}

func init() {
	astCmd.Flags().IntVarP(&ast.TopK, "top", "t", 20, "Number of top elements to display in chart")
	astCmd.Flags().BoolVarP(&ast.Web, "web", "w", false, "Show report in web page")
	astCmd.Flags().BoolVarP(&ast.GeneratePrompt, "generate-prompt", "g", false, "Generate LLM Prompt for further insight")
	astCmd.Flags().BoolVarP(&color.NoColor, "no-color", "n", false, "Disable colorized output")
}
