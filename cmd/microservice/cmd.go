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

The consolidation process, involves the following key components:

1. Define Manifest       ┐
2. Generate Starter      │
   & User Custom Code    ├─> federate ─> Optimized Target System ─> Deployed Application
3. Microservices         │
   (Multiple Systems)    ┘

1. Manifest: Acts as a specification, describing how to generate the target system. Similar to Kubernetes manifests, 
   it uses declarative programming to define the structure and spec of the target system.
2. Starter Project: Generates necessary underlying Java code for runtime and provides a base for user customization. 
   This forms the foundation upon which the optimized target system is built, serving two key purposes:
   - Enhancing efficiency through pre-generated, optimized code
   - Providing runtime support for conflict resolution and resource loading
3. Microservices: Existing microservice systems to be consolidated into the target system.

The 'federate' process, analogous to nvcc's compilation phase, performs the following key steps:
- Parsing and analyzing the manifest, code, and Spring-related resources
- Reconciling conflicts between microservices, including but not limited to:
  * Naming conflicts in Spring contexts and configuration files (e.g., bean names, property keys, classpath)
  * Injection conflicts, such as @Resource annotations assuming single bean instances
- Merging duplicate resources and optimizing dependencies 
- Generating an optimized target system

This approach decouples logical boundaries (how code is written) from physical boundaries (how code is deployed), offering
greater flexibility in system design and operation while maintaining the benefits of microservice architecture.`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, merge.MergeCmd, optimizeCmd, explain.CmdGroup)
}
