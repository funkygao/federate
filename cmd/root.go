package cmd

import (
	"log"
	"os"

	"federate/cmd/chatgpt"
	"federate/cmd/debug"
	"federate/cmd/image"
	"federate/cmd/microservice"
	"federate/cmd/plus"
	"federate/cmd/util"
	"federate/cmd/version"
	"federate/cmd/workload"
	"federate/pkg/logo"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "federate",
	Short: "federate - A compiler and toolchain for microservices management",
	Long: logo.Federate() + `: A compiler-centric toolchain engineered for efficient microservices consolidation and seamless deployment.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	log.SetFlags(0) // log.Lshortfile

	cobra.EnableCommandSorting = false

	// 分组设置
	microservice.CmdGroup.GroupID = "microservice"
	plus.CmdGroup.GroupID = "microservice"
	debug.CmdGroup.GroupID = "microservice"
	workload.CmdGroup.GroupID = "microservice"
	image.CmdGroup.GroupID = "microservice"

	chatgpt.CmdGroup.GroupID = "utility"
	util.CmdGroup.GroupID = "utility"

	version.CmdGroup.GroupID = "system"
	rootCmd.SetHelpCommandGroupID("system")
	rootCmd.SetCompletionCommandGroupID("system")

	rootCmd.AddGroup(
		&cobra.Group{
			ID:    "microservice",
			Title: "Microservice Commands:",
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

	// utility
	rootCmd.AddCommand(chatgpt.CmdGroup, util.CmdGroup)

	// microservice, sorted
	rootCmd.AddCommand(microservice.CmdGroup)
	rootCmd.AddCommand(debug.CmdGroup)
	rootCmd.AddCommand(workload.CmdGroup)
	rootCmd.AddCommand(plus.CmdGroup)
	rootCmd.AddCommand(image.CmdGroup)

	// system
	rootCmd.AddCommand(version.CmdGroup)
}
