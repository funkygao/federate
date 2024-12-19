package tabular

import (
	"log"
	"sort"

	"federate/pkg/util"
	"github.com/olekukonko/tablewriter"
)

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

	// Calculate column widths
	termWidth := util.TerminalWidth()
	colCount := len(header)
	availableWidth := termWidth - (colCount + 1) // Subtract border characters
	colWidths := calculateColumnWidths(header, data, colCount, availableWidth)

	// Set column alignments based on calculated widths
	alignments := make([]int, colCount)
	for i := range alignments {
		alignments[i] = tablewriter.ALIGN_LEFT
	}
	table.SetColumnAlignment(alignments)

	// Disable auto-formatting for headers
	table.SetAutoFormatHeaders(false)

	for _, row := range data {
		newRow := make([]string, colCount)
		copy(newRow, row)
		// Truncate only the last column if necessary
		newRow[colCount-1] = util.Truncate(row[colCount-1], colWidths[colCount-1])
		table.Append(newRow)
	}

	table.Render()
}

func calculateColumnWidths(header []string, data [][]string, colCount, availableWidth int) []int {
	colWidths := make([]int, colCount)
	maxWidths := make([]int, colCount)

	// Calculate max width for each column
	for col := 0; col < colCount; col++ {
		maxWidths[col] = util.TerminalDisplayWidth(header[col])
		for _, row := range data {
			width := util.TerminalDisplayWidth(row[col])
			if width > maxWidths[col] {
				maxWidths[col] = width
			}
		}
	}

	// Distribute available width
	totalMaxWidth := 0
	for _, width := range maxWidths {
		totalMaxWidth += width
	}

	if totalMaxWidth <= availableWidth {
		// If total max width is less than available width, use max widths
		copy(colWidths, maxWidths)
	} else {
		// Otherwise, distribute width proportionally
		for col := 0; col < colCount-1; col++ {
			colWidths[col] = maxWidths[col] * availableWidth / totalMaxWidth
			availableWidth -= colWidths[col]
		}
		// Assign remaining width to the last column
		colWidths[colCount-1] = availableWidth
	}

	return colWidths
}
