package ast

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"github.com/fatih/color"
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
	color.Magenta("Top Complex Loop Structures: %d", TopK)
	tabular.Display([]string{"File", "Method", "Line", "Loop Type", "Nested", "Body"}, cellData, false, -1)
}