package cmd

import (
	"fmt"
	"log"
	"os"

	"federate/cmd/chatgpt"
	"federate/cmd/github"
	"federate/cmd/image"
	"federate/cmd/microservice"
	"federate/cmd/onpremise"
	"federate/cmd/version"
	"federate/pkg/logo"

	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "federate",
		Short: "federate - A compiler and toolchain for microservices management",
		Long: logo.Federate() + `: A cutting-edge, compiler-centric toolchain engineered for efficient microservices consolidation and seamless deployment.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
	}

	allCmd = &cobra.Command{
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
)

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func listSubcommands(cmd *cobra.Command, indent string) {
	fmt.Printf("%s%s - %s\n", indent, cmd.Use, cmd.Short)

	for _, subCmd := range cmd.Commands() {
		if !subCmd.IsAvailableCommand() || subCmd.IsAdditionalHelpTopicCommand() {
			continue
		}
		listSubcommands(subCmd, indent+"  ")
	}
}

func init() {
	log.SetFlags(0) // log.Lshortfile

	cobra.EnableCommandSorting = false

	// 分组设置
	microservice.CmdGroup.GroupID = "microservice"
	onpremise.CmdGroup.GroupID = "microservice"
	image.CmdGroup.GroupID = "microservice"

	ygrepCmd.GroupID = "utility"
	chatgpt.CmdGroup.GroupID = "utility"
	github.CmdGroup.GroupID = "utility"
	allCmd.GroupID = "utility"
	componentsCmd.GroupID = "utility"
	inventoryCmd.GroupID = "utility"

	version.CmdGroup.GroupID = "system"
	rootCmd.SetHelpCommandGroupID("system")
	rootCmd.SetCompletionCommandGroupID("system")

	rootCmd.AddGroup(
		&cobra.Group{
			ID:    "microservice",
			Title: "Microservice Management Commands:",
		},
		&cobra.Group{
			ID:    "utility",
			Title: "Utility Commands:",
		},
		&cobra.Group{
			ID:    "system",
			Title: "System and Help Commands:",
		},
	)

	rootCmd.AddCommand(chatgpt.CmdGroup, ygrepCmd, github.CmdGroup, inventoryCmd, componentsCmd, allCmd)
	rootCmd.AddCommand(microservice.CmdGroup, image.CmdGroup, onpremise.CmdGroup)
	rootCmd.AddCommand(version.CmdGroup)
}
