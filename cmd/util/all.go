package util

import (
	"fmt"

	"github.com/spf13/cobra"
)

var allCmd = &cobra.Command{
	Use:   "all",
	Short: "List all subcommands recursively",
	Long: `The 'all' command lists all subcommands recursively.

It displays the entire command tree, showing the hierarchy of all
available commands and subcommands.`,

	Run: func(cmd *cobra.Command, args []string) {
		// Get the root command
		root := cmd.Root()
		// List all subcommands recursively
		listSubcommands(root, "")
	},
}

func listSubcommands(cmd *cobra.Command, indent string) {
	fmt.Printf("%s%s - %s\n", indent, cmd.Use, cmd.Short)

	for _, subCmd := range cmd.Commands() {
		if !subCmd.IsAvailableCommand() || subCmd.IsAdditionalHelpTopicCommand() {
			continue
		}
		listSubcommands(subCmd, indent+"    ")
	}
}
