package ast

import (
	"fmt"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"github.com/fatih/color"
)

func (i *Info) showCompositionReport() {
	// 统计每个类被组合的次数，同时过滤掉忽略的类型
	compositionCounts := make(map[string]int)
	for _, comp := range i.Compositions {
		compositionCounts[comp.ComposedClass]++
	}

	// 转换为切片并排序
	type compositionCount struct {
		className string
		count     int
	}
	var sortedCounts []compositionCount
	for class, count := range compositionCounts {
		sortedCounts = append(sortedCounts, compositionCount{class, count})
	}
	sort.Slice(sortedCounts, func(i, j int) bool {
		return sortedCounts[i].count > sortedCounts[j].count
	})

	var cellData [][]string
	for _, cc := range sortedCounts[:min(TopK, len(sortedCounts))] {
		var sampleCompositions []string
		for _, comp := range i.Compositions {
			if comp.ComposedClass == cc.className {
				sampleCompositions = append(sampleCompositions, comp.ContainingClass)
				if len(sampleCompositions) == 3 {
					break
				}
			}
		}
		cellData = append(cellData, []string{
			cc.className,
			fmt.Sprintf("%d", cc.count),
			strings.Join(sampleCompositions, ", "),
		})
	}

	color.Magenta("Top Class Compositions: %d", TopK)
	tabular.Display([]string{"Composed Class", "Usage Count", "Sample Containing Classes"}, cellData, false, -1)
}
