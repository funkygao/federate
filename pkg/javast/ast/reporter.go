package ast

import (
	"fmt"
	"log"

	"federate/internal/fs"
	"federate/pkg/primitive"
	"federate/pkg/tabular"
	"github.com/fatih/color"
)

func (i *Info) ShowReport() {
	if Verbosity > 2 {
		i.showNameCountSection("Annotations", []string{"Annotation"}, topN(i.Annotations, TopK))
		i.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	}

	log.Println()
	i.showNameCountSection("Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	log.Println()
	i.showNameCountSection("Variables", []string{"Declaration", "Reference"}, topN(i.Variables, TopK), topN(i.VariableReferences, TopK))
	log.Println()

	i.showInterfacesReport()
	log.Println()
	i.showInheritanceReport()
	log.Println()
	i.showComplexConditionsReport()
	log.Println()
	i.showCompositionReport()
	log.Println()
	i.showComplexLoopsReport()
	log.Println()
	i.showFunctionalUsageReport()

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, variable references: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.VariableReferences), len(i.MethodCalls))

	if false && fs.IsRunningInITerm2() {
		log.Println()
		fs.DisplayJPGInITerm2("templates/arch_layer.jpg")
	}
}

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
