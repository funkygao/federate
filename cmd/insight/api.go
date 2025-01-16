package insight

import (
	"log"

	"federate/pkg/javast"
	"federate/pkg/javast/api"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var apiCmd = &cobra.Command{
	Use:   "api <dir>",
	Short: "Mining Java RPC API Declarations from a Java source directory",
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		runAPICommand(root)
	},
}

func runAPICommand(root string) {
	driver := javast.NewJavastDriver().Verbose()
	info, err := driver.ExtractAPI(root)
	if err != nil {
		log.Fatalf("%s %v", root, err)
	}

	info.ShowReport()
}

func init() {
	apiCmd.Flags().BoolVarP(&api.GeneratePrompt, "generate-prompt", "g", false, "Generate LLM Prompt for further insight")
	apiCmd.Flags().BoolVarP(&color.NoColor, "no-color", "n", false, "Disable colorized output")

	if api.GeneratePrompt {
		color.NoColor = true
	}
}
