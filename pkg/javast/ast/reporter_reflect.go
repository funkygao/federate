package ast

import (
	"fmt"
	"sort"

	"federate/pkg/tabular"
)

func (i *Info) showReflectionReport() (empty bool) {
	// 按类型统计反射使用
	typeCount := make(map[string]int)
	for _, usage := range i.ReflectionUsages {
		typeCount[usage.Type]++
	}

	if len(typeCount) > 1 {
		i.showNameCountSection("Reflection Usage by Type", []string{"Type"}, topN(mapToNameCount(typeCount), TopK))
	}

	// 显示最频繁的反射调用
	sort.Slice(i.ReflectionUsages, func(a, b int) bool {
		return i.ReflectionUsages[a].Name < i.ReflectionUsages[b].Name
	})

	nameCount := make(map[string]int)
	for _, usage := range i.ReflectionUsages {
		nameCount[usage.Name]++
	}

	i.showNameCountSection("Most Frequent Reflection Calls", []string{"Call"}, topN(mapToNameCount(nameCount), TopK))

	// 显示反射使用的详细信息
	if Verbosity > 1 {
		n := min(TopK, len(i.ReflectionUsages))
		var cellData [][]string
		for _, usage := range i.ReflectionUsages[:n] {
			cellData = append(cellData, []string{
				usage.Type,
				usage.Name,
				usage.Location,
				fmt.Sprintf("%d", usage.LineNumber),
			})
		}

		i.writeSectionHeader("%d Detailed Reflection Usage:", n)
		i.writeSectionBody(func() {
			tabular.Display([]string{"Type", "Name", "Location", "Line"}, cellData, false, -1)
		})
	}

	return
}
