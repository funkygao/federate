package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
	Long: `The microservice command group functions as a specialized 'immunosuppressive compiler', employing heuristic 
rule-based approach to seamlessly consolidate multiple microservices into a single JVM deployment. Users simply 
author declarative manifests, while the system autonomously manages the 'transplantation process' - reconciling
both direct and indirect 'tissue rejections' through advanced code rewriting, a two-phase generation process, 
and pre-runtime JAR compatibility analysis. This approach effectively optimizes inter-service communication and 
ensures harmonious integration of diverse service components.

This approach decouples logical boundaries from physical deployment, offering flexibility while maintaining
microservice architecture benefits. The result is a system combining microservice agility with consolidated
architecture efficiency.`,
}

func init() {
	CmdGroup.AddCommand(monolithCmd, fusionStartCmd, componentsCmd, merge.MergeCmd, optimizeCmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
