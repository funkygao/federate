package chatgpt

import (
	"federate/pkg/prompt"
	"github.com/spf13/cobra"
)

var promptCmd = &cobra.Command{
	Use:   "prompt",
	Short: "Interactively generate ChatGPT prompt like using Cursor AI IDE",
	Long: `The prompt command interactively generates ChatGPT prompt like using Cursor AI IDE.

It supports the @Files and @Folders mention mechanism.

Example usage:
  federate chatgpt prompt [codebase dir]`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		codebaseDir := "."
		if len(args) > 0 {
			codebaseDir = args[0]
		}
		generatePrompt(codebaseDir)
	},
}

func generatePrompt(codebaseDir string) {
	prompt.Interact(codebaseDir)
}

func init() {
	promptCmd.Flags().BoolVarP(&prompt.Echo, "echo", "e", true, "Echo mentioned file contents")
	promptCmd.Flags().BoolVarP(&prompt.Dump, "dump", "d", false, "Dump prompt to file")
}
