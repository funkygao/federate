package ast

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showComplexConditionsReport() {
	sort.Slice(i.ComplexConditions, func(a, b int) bool {
		return i.ComplexConditions[a].Complexity > i.ComplexConditions[b].Complexity
	})

	var cellData [][]string
	for _, condition := range i.ComplexConditions {
		cellData = append(cellData, []string{
			strings.Trim(filepath.Base(condition.FileName), ".java"),
			condition.MethodName,
			fmt.Sprintf("%d", condition.LineNumber),
			condition.Condition,
			fmt.Sprintf("%d", condition.Complexity),
		})
	}

	i.writeSectionHeader("Complex Conditions:")
	i.writeSectionBody(func() {
		tabular.Display([]string{"File", "Method", "Line", "Condition", "Score"}, cellData, false, -1)
	})
}
