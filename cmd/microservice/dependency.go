package microservice

import (
	"log"

	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var dependencyCmd = &cobra.Command{
	Use:   "dependency",
	Short: "Compare the target dependency tree with its components",
	Long: `The validate command compares the target dependency tree with its components.

Example usage:
  federate microservice dependency -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m, err := manifest.LoadManifest()
		if err != nil {
			log.Fatalf("Error loading manifest: %v", err)
		}

		checkdependency(m)
	},
}

func checkdependency(m *manifest.Manifest) {
}

func init() {
	manifest.RequiredManifestFileFlag(dependencyCmd)
}
