package cmd

import (
	"fmt"
	"log"
	"os"

	"federate/cmd/chatgpt"
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
		Short: "federate - A compiler and toolkit for microservices management",
		Long: logo.Federate() + ` is a compiler-like toolkit for microservices consolidation and deployment.

  This Swiss Army knife for microservices offers a wide range of functionalities to streamline 
  development, deployment, and management processes.

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

	rootCmd.AddCommand(allCmd, ygrepCmd)
	rootCmd.AddCommand(onpremise.CmdGroup, microservice.CmdGroup, version.CmdGroup, chatgpt.CmdGroup, image.CmdGroup)
}
