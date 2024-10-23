package snap

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snap",
	Short: "Create a secure, tailored source code snapshot for customer delivery",
	Long: `Snap creates a tailored, secure source code snapshot ready for customer delivery.

Key features:
1. Processes git submodules in-place, enabling easy review via diff
2. Removes sensitive information and unnecessary code
3. Eliminates configuration files unrelated to the target profile
4. Ensures post-delivery compilability, allowing customers to build the code independently

The command balances transparency with proprietary protection by:
- Sanitizing internal references and configurations
- Preparing a local Maven repository with necessary dependencies
- Adjusting build scripts for customer environment compatibility
- Retaining only relevant configuration for the specified profile

Outputs:
1. Source Code Repository:
   A git repository tailored for the target environment, with sensitive
   information and irrelevant configurations removed.

2. Local Maven Repository:
   A collection of JAR files and dependencies required for compilation,
   without exposing proprietary source code.

These outputs form a professional-grade, ready-to-deliver package that
enables customers to build and run the application independently.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.LoadManifest()
		runSnap(m)
	},
}

func init() {
	manifest.RequiredManifestFileFlag(Cmd)
}
