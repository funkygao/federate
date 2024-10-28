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
	Long: `Streamline and enhance your microservice architecture through intelligent consolidation and optimization.

To initiate a fusion project:
1. Create a new repository and author a manifest.yaml file.
2. Navigate to the repository and run 'federate microservice scaffold' to generate 
   the project structure and essential files (e.g., Makefile, Dockerfile).
3. Execute 'make consolidate' to merge and optimize microservices.
4. Push changes to the repository to trigger deployment.`,
}

func init() {
	CmdGroup.AddCommand(monolithCmd, fusionStartCmd, merge.MergeCmd, optimize.CmdGroup, manifestCmd)
	if false {
		CmdGroup.AddCommand(explain.CmdGroup)
	}
}
