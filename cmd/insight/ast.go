package insight

import (
	"log"

	"federate/pkg/javast"
	"federate/pkg/javast/ast"
	"github.com/spf13/cobra"
)

var astCmd = &cobra.Command{
	Use:   "ast <dir>",
	Short: "Extract AST information from a Java source directory",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		runASTCommand(args[0])
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
}
