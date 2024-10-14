package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
}

func init() {
	CmdGroup.AddCommand(monolithCmd, fusionStartCmd, merge.MergeCmd, optimizeCmd, manifestCmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
