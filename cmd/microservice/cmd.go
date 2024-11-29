package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/microservice/optimize"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
	Long:  `Streamline and enhance your microservice architecture through strategic consolidation and optimization.`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, fusionStartCmd, mergeCmd, optimize.CmdGroup, jdosCmd, manifestCmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
