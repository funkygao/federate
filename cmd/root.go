package cmd

import (
	"log"
	"os"

	"federate/cmd/chatgpt"
	"federate/cmd/debug"
	"federate/cmd/image"
	"federate/cmd/insight"
	"federate/cmd/microservice"
	"federate/cmd/plus"
	"federate/cmd/util"
	"federate/cmd/version"
	"federate/cmd/workload"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var logo = color.New(color.Bold).Add(color.Underline).Add(color.FgCyan).Sprintf("federate")

var rootCmd = &cobra.Command{
	Use:   "federate",
	Short: "federate - A compiler and toolchain for microservices management",
	Long: logo + `: A compiler-centric toolchain engineered for efficient microservices consolidation and seamless deployment.

  Find more information at: https://joyspace.jd.com/pages/Ksl7N7wr1XxFanCRIR1y
`,
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	log.SetFlags(0)          // log.Lshortfile
	log.SetOutput(os.Stdout) // 默认 stderr

	cobra.EnableCommandSorting = false

	// 分组设置
	microservice.CmdGroup.GroupID = "microservice"
	plus.CmdGroup.GroupID = "microservice"
	workload.CmdGroup.GroupID = "microservice"

	debug.CmdGroup.GroupID = "utility"
	image.CmdGroup.GroupID = "utility"
	util.CmdGroup.GroupID = "utility"

	chatgpt.CmdGroup.GroupID = "explore"
	insight.CmdGroup.GroupID = "explore"

	version.CmdGroup.GroupID = "system"
	rootCmd.SetHelpCommandGroupID("system")
	rootCmd.SetCompletionCommandGroupID("system")

	rootCmd.AddGroup(
		&cobra.Group{
			ID:    "microservice",
			Title: "Microservice Commands:",
		},
		&cobra.Group{
			ID:    "explore",
			Title: "Exploration Commands:",
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
	rootCmd.AddCommand(chatgpt.CmdGroup, insight.CmdGroup, image.CmdGroup, debug.CmdGroup, util.CmdGroup)

	// microservice, sorted
	rootCmd.AddCommand(microservice.CmdGroup)
	rootCmd.AddCommand(plus.CmdGroup)
	rootCmd.AddCommand(workload.CmdGroup)

	// system
	rootCmd.AddCommand(version.CmdGroup)
}
