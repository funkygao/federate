package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Commands for managing the lifecycle of microservices",
	Long: `The microservice command group manages the lifecycle of microservices, addressing issues of granularity and facilitating consolidation.

Key benefits:
- Improved performance through reduced class loading and conversion of RPC to JVM calls
- Enhanced resource efficiency and reduced duplicate resource consumption across services
- Higher deployment density and lower operational costs
- Simplified service interactions and reduced complexity in service chains

This approach decouples logical boundaries (how code is written) from physical boundaries (how code is deployed), offering
greater flexibility in system design and operation.

wms-microfusion acts as logical monoliths, offload the decisions of how to distribute and run applications to federate runtime.`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, merge.MergeCmd, optimizeCmd, validateCmd, dependencyCmd, explain.CmdGroup)
}
