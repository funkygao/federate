package plus

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Generate a new plus project with standard structure",
	Long:  `Scaffold a new plus project, laying the foundation for platform extensions with an optimized directory structure and essential boilerplate code.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doCreate(m)
	},
}

func doCreate(m *manifest.Manifest) {
}

func init() {
	manifest.RequiredManifestFileFlag(createCmd)
}
