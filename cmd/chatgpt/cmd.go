package chatgpt

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "chatgpt",
	Short: "Commands for managing chatgpt prompt generation",
	Long:  `The chatgpt command group provides a set of commands to manage chatgpt prompt generation.`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			promptCmd.Run(cmd, args)
		}
	},
}

func init() {
	CmdGroup.AddCommand(promptCmd, tokensCmd)
}
