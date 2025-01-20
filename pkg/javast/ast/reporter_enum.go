package ast

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"federate/pkg/tabular"
	"federate/pkg/util"
)

func (i *Info) showEnumReport() (empty bool) {
	if len(i.Enums) == 0 {
		return true
	}

	enumCounts := make(map[string]int) // 统计 enum 名称出现次数
	maxWidth := 0
	for _, enum := range i.Enums {
		maxWidth = max(maxWidth, util.TerminalDisplayWidth(enum.Name))
		enumCounts[enum.Name]++
	}

	var cellData [][]string
	for _, enum := range i.Enums {
		sort.Strings(enum.Values)
		cellData = append(cellData, []string{
			enum.Name,
			fmt.Sprintf("%d", len(enum.Values)),
			util.Truncate(strings.Join(enum.Values, "/"), util.TerminalWidth()-maxWidth-15),
		})
	}

	// 对 cellData 按 Enum Name 排序
	sort.Slice(cellData, func(i, j int) bool {
		return cellData[i][0] < cellData[j][0]
	})

	i.writeSectionHeader("%d Enum Analysis", len(i.Enums))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Enum Name", "N", "Values"}, cellData, false, -1)
	})

	i.showDupEnumReport(enumCounts)

	return
}

func (i *Info) showDupEnumReport(enumCounts map[string]int) {
	var duplicateEnums [][]string

	for name, count := range enumCounts {
		if count > 1 {
			duplicateEnums = append(duplicateEnums, []string{
				name,
				fmt.Sprintf("%d", count),
			})
		}
	}

	if len(duplicateEnums) > 0 {
		// 对重复的 enums 按出现次数降序排序，次数相同时按名称升序排序
		sort.Slice(duplicateEnums, func(i, j int) bool {
			countI, _ := strconv.Atoi(duplicateEnums[i][1])
			countJ, _ := strconv.Atoi(duplicateEnums[j][1])
			if countI != countJ {
				return countI > countJ // 降序排序
			}
			return duplicateEnums[i][0] < duplicateEnums[j][0] // 名称升序排序
		})

		i.writeSectionHeader("%d Duplicate Enum Definitions", len(duplicateEnums))
		i.writeSectionBody(func() {
			tabular.Display([]string{"Enum Name", "Occurrences"}, duplicateEnums, false, -1)
		})
	}
}
