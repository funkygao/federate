package cmd

import (
	"embed"
	"log"
	"os"

	"federate/cmd/onpremise"
	"github.com/fatih/color"
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
		Long: color.New(color.Bold).Sprintf("federate") + ` is a swiss army knife for microservices consolidation based on manifest DSL.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
	}

	microserviceCmdGroup = &cobra.Command{
		Use:   "microservice",
		Short: "Commands for managing the lifecycle of microservices",
		Long: `The microservice command group manages the lifecycle of microservices, addressing issues of granularity and facilitating consolidation.

Key benefits:
- Improved performance through reduced class loading and conversion of RPC to JVM calls
- Enhanced resource efficiency and reduced duplicate resource consumption across services
- Higher deployment density and lower operational costs
- Simplified service interactions and reduced complexity in service chains

This approach decouples logical boundaries (code development) from physical boundaries (deployment strategy) of
microservices, offering greater flexibility in system design and operation.`,
	}

	explainCmdGroup = &cobra.Command{
		Use:   "explain",
		Short: "Describes microservice fusion key mechanisms",
		Long:  `The explain command describes microservice fusion key mechanisms`,
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

	// root
	rootCmd.AddCommand(onpremise.CmdGroup, microserviceCmdGroup, versionCmdGroup, chatgptCmdGroup, imageCmdGroup, ygrepCmd)

	// groups
	microserviceCmdGroup.AddCommand(mergeCmd, conventionCmd, optimizeCmd, validateCmd, explainCmdGroup)
	chatgptCmdGroup.AddCommand(promptCmd, tokensCmd)
	explainCmdGroup.AddCommand(manifestCmd, taintCmd, assumptionCmd)
	versionCmdGroup.AddCommand(upgradeCmd, versionCmd)
	imageCmdGroup.AddCommand(buildRpmCmd, buildDockerCmd)
}
