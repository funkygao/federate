package plus

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var bundleCmd = &cobra.Command{
	Use:   "bundle",
	Short: "Package the plus project for distribution",
	Long:  `Bundle the plus project with dependencies and resources for streamlined deployment and seamless integration.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doBundle(m)
	},
}

func doBundle(m *manifest.Manifest) {
}

func init() {
	manifest.RequiredManifestFileFlag(bundleCmd)
}
