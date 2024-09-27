package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
	Long: `The microservice command group streamlines the consolidation and optimization of multiple microservices 
into a cohesive, efficient system while preserving the flexibility of the microservice architecture.

The consolidation process involves the following key components:

1. Define Manifest       ┐
2. Generate Starter      │
   & User Custom Code    ├─> federate ─> Optimized Target System ─> Deployed Application
3. Microservices         │
   (Multiple Systems)    ┘

1. Manifest: Specifies the structure and configuration of the target system using declarative programming.
2. Starter Project: Provides optimized runtime code, conflict resolution support, and a customizable foundation.
3. Microservices: Existing independent systems to be consolidated and optimized as a cohesive unit.

The 'federate' process performs key steps including:
- Analyzing manifests, code, and resources
- Reconciling conflicts (e.g., naming, injection, dependencies)
- Merging resources and optimizing dependencies
- Generating an optimized target system

This approach decouples logical boundaries from physical deployment, offering flexibility while maintaining
microservice architecture benefits. The result is a system combining microservice agility with consolidated
architecture efficiency.`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, merge.MergeCmd, optimizeCmd, explain.CmdGroup)
}
