package explain

import (
	"sort"

	"federate/internal/fs"
	"federate/pkg/convention"
	"federate/pkg/tablerender"
	"github.com/spf13/cobra"
)

var assumptionCmd = &cobra.Command{
	Use:   "assumption",
	Short: "Display federation assumptions in a table format",
	Long: `The assumption command displays all the federation assumptions in a table format.

Example usage:
  federate microservice explain assumption`,
	Run: func(cmd *cobra.Command, args []string) {
		displayAssumptions()
	},
}

func displayAssumptions() {
	allAssumptions := convention.GetAll(fs.FS, "templates/doc/assumptions.yml")
	kinds := allAssumptions.Kinds()

	// Collect all assumptions into a slice for sorting
	var assumptions [][]string
	for _, kind := range kinds {
		keys := allAssumptions.GetKeys(kind)
		sort.Strings(keys)
		for _, key := range keys {
			example := allAssumptions.GetExample(kind, key)
			assumptions = append(assumptions, []string{kind, key, example})
		}
	}

	header := []string{"Category", "Key", "Value Convention"}
	tablerender.DisplayTable(header, assumptions, true, 0, 1)
}
