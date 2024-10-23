package version

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "version",
	Short: "Commands for managing version of the federate tool",
	Long:  `The version command group provides a set of commands to manage version of the federate tool`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			versionCmd.Run(cmd, args)
		}
	},
}

func init() {
	CmdGroup.AddCommand(upgradeCmd, versionCmd)
}
