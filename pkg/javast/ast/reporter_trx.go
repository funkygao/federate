package ast

import (
	"path/filepath"
	"strconv"
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showTransactionReport() (empty bool) {
	if len(i.TransactionInfos) == 0 {
		return true
	}

	var cellData [][]string
	for _, info := range i.TransactionInfos {
		cellData = append(cellData, []string{
			strings.Trim(filepath.Base(info.FileName), ".java"),
			info.MethodName,
			strconv.Itoa(info.LineNumber),
			info.Type,
		})
	}

	i.writeSectionHeader("%d Transactions", len(cellData))
	i.writeSectionBody(func() {
		tabular.Display([]string{"File Name", "Method Name", "Line Number", "Type"}, cellData, true, -1)
	})

	return
}
