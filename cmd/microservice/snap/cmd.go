package snap

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snap",
	Short: "Create a secure, tailored source code snapshot for customer delivery",
	Long: `Snap creates a secure, tailored source code snapshot for customer delivery.

This command:
1. Processes each component as a git submodule
2. Performs in-place updates to maintain git diff functionality
3. Removes sensitive information and unnecessary code
4. Prepares a local Maven repository for customer use
5. Ensures the final package is compilable in the customer's environment

The resulting snapshot is a professional-grade, ready-to-deliver source code snapshot
that balances transparency with the protection of proprietary information.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.LoadManifest()
		runSnap(m)
	},
}

func init() {
	manifest.RequiredManifestFileFlag(Cmd)
}
