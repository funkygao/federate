package ast

import (
	"log"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"github.com/fatih/color"
)

func (i *Info) ShowReport() {
	filteredInfo := i.ApplyFilters(
		&IgnoreInterfacesFilter{IgnoredInterfaces: ignoredInterfaces},
		&IgnoreAnnotationsFilter{IgnoredAnnotations: ignoredAnnotations},
	)

	filteredInfo.showInheritanceReport()

	filteredInfo.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	filteredInfo.showNameCountSection("Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	filteredInfo.showNameCountSection("Variables", []string{"Declaration", "Reference"}, topN(i.Variables, TopK), topN(i.VariableReferences, TopK))
	filteredInfo.showNameCountSection("Annotations", []string{"Annotation"}, topN(i.Annotations, TopK))
	filteredInfo.showInterfacesReport()

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, variable references: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.VariableReferences), len(i.MethodCalls))
}

func (i *Info) showInheritanceReport() {
	color.Magenta("Class Inheritance Hierarchy:")
	var inheritanceData [][]string
	for class, superclasses := range i.Inheritance {
		inheritanceData = append(inheritanceData, []string{class, strings.Join(superclasses, " -> ")})
	}
	sort.Slice(inheritanceData, func(i, j int) bool {
		return inheritanceData[i][0] < inheritanceData[j][0]
	})
	tabular.Display([]string{"Class", "Inheritance"}, inheritanceData, false, -1)
}

func (i *Info) showInterfacesReport() {
	interfaceCounts := make(map[string]int)
	for _, interfaces := range i.Interfaces {
		for _, iface := range interfaces {
			interfaceCounts[iface]++
		}
	}
	i.showNameCountSection("Implemented Interfaces", []string{"Interface"}, topN(mapToNameCount(interfaceCounts), TopK))
}
