package cmd

import (
	"embed"
	"log"
	"os"

	"github.com/spf13/cobra"
)

var (
	manifestFile string

	//go:embed templates/*
	templates embed.FS
)

var (
	rootCmd = &cobra.Command{
		Use:   "federate",
		Short: "federate - A tool for merging and deploying microservices",
		Long: `federate is a Swiss Army knife CLI tool for merging and managing microservices based on manifest DSL.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
	}

	explainCmdGroup = &cobra.Command{
		Use:   "explain",
		Short: "Describes microservice fusion key mechanisms",
		Long:  `The explain command describes microservice fusion key mechanisms`,
	}

	microserviceCmdGroup = &cobra.Command{
		Use:   "microservice",
		Short: "Commands for managing the lifecycle of microservices",
		Long:  `The microservice command group provides a set of commands to manage the lifecycle of microservices.`,
	}

	chatgptCmdGroup = &cobra.Command{
		Use:   "chatgpt",
		Short: "Commands for managing chatgpt prompt generation",
		Long:  `The chatgpt command group provides a set of commands to manage chatgpt prompt generation.`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				promptCmd.Run(cmd, args)
			}
		},
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

	imageCmdGroup = &cobra.Command{
		Use:   "image",
		Short: "Commands for managing images like Docker and RPM",
		Long:  `The image command group provides a set of commands to manage images like Docker and RPM.`,
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

	// groups
	microserviceCmdGroup.AddCommand(mergeCmd, conventionCmd, optimizeCmd, validateCmd, explainCmdGroup)
	chatgptCmdGroup.AddCommand(promptCmd, tokensCmd)
	explainCmdGroup.AddCommand(taintCmd, assumptionCmd)
	versionCmdGroup.AddCommand(upgradeCmd, versionCmd)
	imageCmdGroup.AddCommand(buildRpmCmd, buildDockerCmd)
	// root
	rootCmd.AddCommand(abcCmd, microserviceCmdGroup, versionCmdGroup, chatgptCmdGroup, imageCmdGroup, ygrepCmd)
}
