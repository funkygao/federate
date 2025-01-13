package ast

import (
	"fmt"

	"federate/pkg/primitive"
	"federate/pkg/tabular"
	"github.com/fatih/color"
)

func (i *Info) showNameCountSection(title string, namesHeader []string, nameCounts ...[]primitive.NameCount) {
	var headers []string
	for _, name := range namesHeader {
		headers = append(headers, name, "Count")
	}

	maxLen := 0
	for _, nc := range nameCounts {
		if len(nc) > maxLen {
			maxLen = len(nc)
		}
	}

	cellData := make([][]string, maxLen)
	for i := range cellData {
		cellData[i] = make([]string, len(headers))
	}

	for colIndex, nc := range nameCounts {
		for rowIndex, item := range nc {
			if rowIndex < maxLen {
				cellData[rowIndex][colIndex*2] = item.Name
				cellData[rowIndex][colIndex*2+1] = fmt.Sprintf("%d", item.Count)
			}
		}
	}

	color.Magenta("Top %s: %d", title, TopK)
	tabular.Display(headers, cellData, false, -1)
}
