package plus

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var assembleCmd = &cobra.Command{
	Use:   "assemble",
	Short: "Create deployable artifact for the Plus project",
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doAssemble(m)
	},
}

func doAssemble(m *manifest.Manifest) {
}

func init() {
	manifest.RequiredManifestFileFlag(assembleCmd)
}
