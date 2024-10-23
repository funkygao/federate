package explain

import (
	"sort"

	"federate/internal/fs"
	"federate/pkg/convention"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var conventionCmd = &cobra.Command{
	Use:   "convention",
	Short: "Display development conventions in a table format",
	Long: `The convention command displays all the development conventions in a table format.
It includes all the specified kinds and their keys with example values.

Example usage:
  federate microservice convention`,
	Run: func(cmd *cobra.Command, args []string) {
		displayConventions()
	},
}

func displayConventions() {
	allConventions := convention.GetAll(fs.FS, "templates/conventions.yml")
	kinds := allConventions.Kinds()

	if len(kinds) == 0 {
		color.Red("No conventions found.")
		return
	}

	// Collect all conventions into a slice for sorting
	var conventions [][]string
	for _, kind := range kinds {
		keys := allConventions.GetKeys(kind)
		sort.Strings(keys)
		for _, key := range keys {
			example := allConventions.GetExample(kind, key)
			conventions = append(conventions, []string{kind, key, example})
		}
	}

	c := color.New(color.FgRed)
	c = c.Add(color.Bold)
	c.Println("One namespace: (appName, classPath, servlet, spring bean registry)")

	header := []string{"Category", "Key", "Value Convention"}
	tablerender.DisplayTable(header, conventions, true, 0, 1)
}
