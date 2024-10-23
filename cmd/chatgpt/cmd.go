package chatgpt

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "chatgpt",
	Short: "Commands for managing ChatGPT Prompt like using Cursor AI IDE",
	Long:  `The chatgpt command group provides a set of commands to manage ChatGPT Prompt ike using Cursor AI IDE`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			promptCmd.Run(cmd, args)
		}
	},
}

func init() {
	CmdGroup.AddCommand(promptCmd, tokensCmd)
}
