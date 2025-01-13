package ast

import (
	"fmt"
	"log"

	"federate/pkg/tabular"
)

var TopK int

func (i *Info) ShowReport() {
	log.Println("Top Items Report:")

	headers := []string{"Import", "Count", "Method", "Count", "Variable", "Count", "Method Call", "Count"}
	var cellData [][]string

	imports := topN(i.Imports, TopK)
	methods := topN(i.Methods, TopK)
	variables := topN(i.Variables, TopK)
	methodCalls := topN(i.MethodCalls, TopK)

	maxRows := max(len(imports), len(methods), len(variables), len(methodCalls))

	for row := 0; row < maxRows; row++ {
		rowData := make([]string, 8)
		if row < len(imports) {
			rowData[0] = imports[row].Name
			rowData[1] = fmt.Sprintf("%d", imports[row].Count)
		}
		if row < len(methods) {
			rowData[2] = methods[row].Name
			rowData[3] = fmt.Sprintf("%d", methods[row].Count)
		}
		if row < len(variables) {
			rowData[4] = variables[row].Name
			rowData[5] = fmt.Sprintf("%d", variables[row].Count)
		}
		if row < len(methodCalls) {
			rowData[6] = methodCalls[row].Name
			rowData[7] = fmt.Sprintf("%d", methodCalls[row].Count)
		}
		cellData = append(cellData, rowData)
	}

	tabular.Display(headers, cellData, true, -1)

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.MethodCalls))
}

func max(nums ...int) int {
	if len(nums) == 0 {
		return 0
	}
	max := nums[0]
	for _, num := range nums[1:] {
		if num > max {
			max = num
		}
	}
	return max
}
