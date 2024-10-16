package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"federate/cmd/microservice/optimize"
	"federate/cmd/microservice/snap"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
}

func init() {
	CmdGroup.AddCommand(monolithCmd, fusionStartCmd, merge.MergeCmd, optimize.CmdGroup, manifestCmd, snap.Cmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
