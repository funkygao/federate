package version

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	GitUser    = "unknown"
	GitCommit  = "unknown"
	GitBranch  = "unknown"
	GitState   = "unknown"
	BuildDate  = "unknown"
	gitSummary = "unknown"
)

var versionCmd = &cobra.Command{
	Use:   "ver",
	Short: "Display the current versionof the federate tool",
	Long: `The version command shows the current version of the federate tool.

Example usage:
  federate version ver`,
	Run: func(cmd *cobra.Command, args []string) {
		showVersion()
	},
}

func showVersion() {
	// Combine Git information into GitSummary
	if GitCommit != "unknown" && GitBranch != "unknown" && GitState != "unknown" {
		gitSummary = fmt.Sprintf("%s-%s-%s", GitCommit, GitBranch, GitState)
	}

	fmt.Printf("Git Summary: %s\n", gitSummary)
	fmt.Printf("Build By: %s\n", GitUser)
	fmt.Printf("Build Date : %s\n", BuildDate)
}
