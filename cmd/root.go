package cmd

import (
	"log"
	"os"

	"federate/cmd/chatgpt"
	"federate/cmd/image"
	"federate/cmd/microservice"
	"federate/cmd/onpremise"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "federate",
		Short: "federate - A tool for merging and deploying microservices",
		Long: color.New(color.Bold).Sprintf("federate") + ` is a swiss army knife for microservices consolidation based on manifest DSL.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
	}

	versionCmdGroup = &cobra.Command{
		Use:   "version",
		Short: "Commands for managing version of the federate tool",
		Long:  `The version command group provides a set of commands to manage version of the federate tool`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				versionCmd.Run(cmd, args)
			}
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

func init() {
	log.SetFlags(0) // log.Lshortfile

	// root
	rootCmd.AddCommand(allCmd, onpremise.CmdGroup, microservice.CmdGroup, versionCmdGroup, chatgpt.CmdGroup, image.CmdGroup, ygrepCmd)

	// groups
	versionCmdGroup.AddCommand(upgradeCmd, versionCmd)
}
