package ast

import (
	"fmt"
	"log"

	"federate/pkg/primitive"
	"federate/pkg/tabular"
)

var TopK int

func (i *Info) ShowReport() {
	log.Println("Top Items Report:")

	i.showNameCountSection("Top Imports", []string{"Import"}, topN(i.Imports, TopK))
	i.showNameCountSection("Top Methods", []string{"Declaration", "Call"}, topN(i.Methods, TopK), topN(i.MethodCalls, TopK))
	i.showNameCountSection("Top Variables", []string{"Declaration"}, topN(i.Variables, TopK))

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.MethodCalls))
}

func (i *Info) showNameCountSection(title string, namesHeader []string, nameCounts ...[]primitive.NameCount) {
	var headers []string
	for _, name := range namesHeader {
		headers = append(headers, []string{name, "Count"}...)
	}

	var cellData [][]string
	for _, rowDatas := range nameCounts {
		var rowData []string
		for _, nc := range rowDatas {
			rowData = append(rowData, []string{nc.Name, fmt.Sprintf("%d", nc.Count)}...)
		}
		log.Printf("%+v", rowData)
		cellData = append(cellData, rowData)
	}
	log.Println(title)
	tabular.Display(headers, cellData, false, -1)
}
