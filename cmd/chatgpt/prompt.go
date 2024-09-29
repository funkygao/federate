package chatgpt

import (
	"federate/pkg/prompt"
	"github.com/spf13/cobra"
)

var echo bool

var promptCmd = &cobra.Command{
	Use:   "prompt",
	Short: "Interactively generate ChatGPT prompt like using Cursor AI IDE",
	Long: `The prompt command interactively generates ChatGPT prompt like using Cursor AI IDE.

It supports the @Files and @Folders mention mechanism.

Example usage:
  federate chatgpt prompt [codebase dir]`,
	Run: func(cmd *cobra.Command, args []string) {
		codebaseDir := "."
		if len(args) > 0 {
			codebaseDir = args[0]
		}
		generatePrompt(codebaseDir)
	},
}

func generatePrompt(codebaseDir string) {
	prompt.Interact(codebaseDir, echo)
}

func init() {
	promptCmd.Flags().BoolVarP(&echo, "echo", "e", true, "Echo mentioned file contents")
}
