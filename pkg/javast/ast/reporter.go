package ast

import (
	"fmt"
	"log"

	"federate/pkg/primitive"
	"federate/pkg/tabular"
	"github.com/fatih/color"
)

var TopK int

func (i *Info) ShowReport() {
	i.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	i.showNameCountSection("Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	i.showNameCountSection("Variables", []string{"Declaration"}, topN(i.Variables, TopK))

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.MethodCalls))
}

func (i *Info) showNameCountSection(title string, namesHeader []string, nameCounts ...[]primitive.NameCount) {
	var headers []string
	for _, name := range namesHeader {
		headers = append(headers, name, "Count")
	}

	// 找出最长的 nameCount 列表
	maxLen := 0
	for _, nc := range nameCounts {
		if len(nc) > maxLen {
			maxLen = len(nc)
		}
	}

	// 准备单元格数据
	cellData := make([][]string, maxLen)
	for i := range cellData {
		cellData[i] = make([]string, len(headers))
	}

	// 填充数据
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
