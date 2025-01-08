package tabular

import (
	"fmt"
	"sort"
	"strings"

	"federate/pkg/util"
	"github.com/fatih/color"
)

type BarChartItem struct {
	Name  string
	Count int
}

func PrintBarChart(items []BarChartItem, topK int) {
	if len(items) == 0 {
		return
	}

	// Sort items by count in descending order
	sort.Slice(items, func(i, j int) bool {
		return items[i].Count > items[j].Count
	})

	// Limit to topK items
	if topK > 0 && topK < len(items) {
		items = items[:topK]
	}

	maxCount := items[0].Count
	maxNameLength := 0
	for _, item := range items {
		if len(item.Name) > maxNameLength {
			maxNameLength = len(item.Name)
		}
	}

	width := util.TerminalWidth()
	barWidth := width - maxNameLength - 15 // 15 for count and spaces
	cyan := color.New(color.FgCyan)
	yellow := color.New(color.FgYellow)

	for i, item := range items {
		barLength := int(float64(item.Count) / float64(maxCount) * float64(barWidth))
		if barLength < 0 {
			barLength = 0
		}
		bar := strings.Repeat("■", barLength)

		fmt.Printf("%-*s %5d ", maxNameLength, item.Name, item.Count)
		if i%2 == 0 {
			cyan.Print(bar)
		} else {
			yellow.Print(bar)
		}
		fmt.Println()
	}
}
