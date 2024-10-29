package cmd

import (
	"fmt"
	"strings"

	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var componentsCmd = &cobra.Command{
	Use:   "components",
	Short: "List components of the manifest",
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.Load()

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
