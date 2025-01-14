package ast

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showComplexLoopsReport() {
	// 按嵌套级别和循环体大小排序
	sort.Slice(i.ComplexLoops, func(a, b int) bool {
		if i.ComplexLoops[a].NestingLevel != i.ComplexLoops[b].NestingLevel {
			return i.ComplexLoops[a].NestingLevel > i.ComplexLoops[b].NestingLevel
		}
		return i.ComplexLoops[a].BodySize > i.ComplexLoops[b].BodySize
	})

	var cellData [][]string
	for _, loop := range i.ComplexLoops[:min(TopK, len(i.ComplexLoops))] {
		cellData = append(cellData, []string{
			strings.Trim(filepath.Base(loop.FileName), ".java"),
			loop.MethodName,
			fmt.Sprintf("%d", loop.LineNumber),
			loop.LoopType,
			fmt.Sprintf("%d", loop.NestingLevel),
			fmt.Sprintf("%d", loop.BodySize),
		})
	}
	i.writeSectionHeader("Top %d Complex Loop Structures", TopK)
	i.writeSectionBody(func() {
		tabular.Display([]string{"File", "Method", "Line", "Loop Type", "Nested", "Body Size"}, cellData, false, -1)
	})
}
