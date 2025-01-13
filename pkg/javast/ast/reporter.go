package ast

import (
	"log"

	"federate/internal/fs"
)

func (i *Info) ShowReport() {
	filteredInfo := i.ApplyFilters(
		&IgnoreInterfacesFilter{IgnoredInterfaces: ignoredInterfaces},
		&IgnoreAnnotationsFilter{IgnoredAnnotations: ignoredAnnotations},
	)

	filteredInfo.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	filteredInfo.showNameCountSection("Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	filteredInfo.showNameCountSection("Variables", []string{"Declaration", "Reference"}, topN(i.Variables, TopK), topN(i.VariableReferences, TopK))
	filteredInfo.showNameCountSection("Annotations", []string{"Annotation"}, topN(i.Annotations, TopK))

	filteredInfo.showInterfacesReport()
	filteredInfo.showInheritanceReport()

	filteredInfo.showComplexConditionsReport()

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, variable references: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.VariableReferences), len(i.MethodCalls))

	if false && fs.IsRunningInITerm2() {
		log.Println()
		fs.DisplayJPGInITerm2("templates/arch_layer.jpg")
	}
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
