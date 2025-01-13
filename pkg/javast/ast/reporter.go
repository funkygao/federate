package ast

import (
	"log"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"github.com/fatih/color"
)

var TopK int

func (i *Info) ShowReport() {
	i.showInheritanceReport()

	i.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	i.showNameCountSection("Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	i.showNameCountSection("Variables", []string{"Declaration"}, topN(i.Variables, TopK))
	i.showNameCountSection("Annotations", []string{"Annotation", "Custom"}, topN(i.Annotations, TopK))
	i.showInterfacesReport()

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.MethodCalls))
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
			if !ignoreInteface(iface) {
				interfaceCounts[iface]++
			}
		}
	}
	i.showNameCountSection("Implemented Interfaces", []string{"Interface"}, topN(mapToNameCount(interfaceCounts), TopK))
}
