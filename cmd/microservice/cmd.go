package microservice

import (
	"federate/cmd/explain"
	"federate/cmd/merge"
	"federate/cmd/microservice/optimize"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "microservice",
	Short: "Orchestrate microservice consolidation and optimization",
	Long: `Streamline and enhance your microservice architecture through strategic consolidation and optimization.

To initiate a fusion project:
1. Create a new repository and author the manifest.yaml file.
2. Execute 'federate microservice scaffold' to generate the project structure 
   and essential files (e.g., Makefile, pom.xml).
3. Execute 'make fusion-start' to generate the fusion-starter Maven module. 
   Use this module as a foundation for further development and customization.
4. Execute 'make consolidate' to merge microservices.`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, fusionStartCmd, merge.MergeCmd, optimize.CmdGroup, manifestCmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
