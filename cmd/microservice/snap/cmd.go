package snap

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snap",
	Short: "Create a tailored source code snapshot for customer delivery",
	Long: `Create a tailored source code snapshot based on the manifest file for specific customer delivery.
This command processes each component as a git submodule, performing in-place updates.
It removes sensitive information and unnecessary code to ensure a secure and optimized delivery.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.LoadManifest()
		runSnap(m)
	},
}

func init() {
	manifest.RequiredManifestFileFlag(Cmd)
}
