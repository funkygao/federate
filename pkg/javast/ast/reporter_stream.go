package ast

import (
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"

	"federate/pkg/tabular"
	"github.com/fatih/color"
)

func (i *Info) showFunctionalUsageReport() {
	color.Magenta("Functional Programming Usage:")

	// 按类型和操作分组
	usageMap := make(map[string]map[string]int)
	for _, usage := range i.FunctionalUsages {
		if _, ok := usageMap[usage.Type]; !ok {
			usageMap[usage.Type] = make(map[string]int)
		}
		usageMap[usage.Type][usage.Operation]++
	}

	// 显示使用统计
	for usageType, operations := range usageMap {
		log.Printf("%s usage:", strings.Title(usageType))
		var cellData [][]string
		for op, count := range operations {
			cellData = append(cellData, []string{op, fmt.Sprintf("%d", count)})
		}
		sort.Slice(cellData, func(i, j int) bool {
			countI, _ := strconv.Atoi(cellData[i][1])
			countJ, _ := strconv.Atoi(cellData[j][1])
			return countI > countJ
		})
		tabular.Display([]string{"Operation", "Count"}, cellData, false, -1)
	}

	// 显示一些示例用法
	log.Println("Sample Usages:")
	var cellData [][]string
	for _, usage := range i.FunctionalUsages[:min(10, len(i.FunctionalUsages))] {
		cellData = append(cellData, []string{
			usage.FileName,
			usage.MethodName,
			usage.Type,
			usage.Operation,
			usage.Context,
			fmt.Sprintf("%d", usage.LineNumber),
		})
	}
	tabular.Display([]string{"File", "Method", "Type", "Operation", "Context", "Line"}, cellData, false, -1)
}
