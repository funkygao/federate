package optimize

import (
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var jarCmd = &cobra.Command{
	Use:   "jar",
	Short: "Optimize project JAR dependencies",
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.LoadManifest()
		optimizeJar(manifest)
	},
}

func optimizeJar(m *manifest.Manifest) {
}

func init() {
	manifest.RequiredManifestFileFlag(jarCmd)
}
