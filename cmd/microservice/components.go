package microservice

import (
	"fmt"
	"strings"

	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var componentsCmd = &cobra.Command{
	Use:   "components",
	Short: "List components of the manifest",
	Long: `The components command lists all components of the manifest.

Example usage:
  federate microservice components -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.LoadManifest()

		listComponents(manifest)
	},
}

func listComponents(m *manifest.Manifest) {
	var components []string
	for _, c := range m.Components {
		components = append(components, c.Name)
	}
	fmt.Println(strings.Join(components, " "))
}

func init() {
	manifest.RequiredManifestFileFlag(componentsCmd)
}
