package tabular

import (
	"log"
	"sort"

	"github.com/olekukonko/tablewriter"
)

// Display displays a table with the given header and data, sorted by the specified columns.
func Display(header []string, data [][]string, autoMergeCells bool, sortByColumns ...int) {
	if len(data) < 1 {
		return
	}

	if len(sortByColumns) > 0 && sortByColumns[0] > -1 {
		// Sort the data based on the specified columns
		sort.Slice(data, func(i, j int) bool {
			for _, col := range sortByColumns {
				if data[i][col] != data[j][col] {
					return data[i][col] < data[j][col]
				}
			}
			return false
		})
	}

	table := tablewriter.NewWriter(log.Default().Writer())
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetAutoMergeCells(autoMergeCells)
	table.SetAutoWrapText(false)
	table.SetHeader(header)

	for _, row := range data {
		table.Append(row)
	}
	table.Render()
}
