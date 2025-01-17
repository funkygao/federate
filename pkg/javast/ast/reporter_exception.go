package ast

import (
	"fmt"
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showExceptionCatches() (empty bool) {
	if len(i.ExceptionCatches) == 0 {
		return true
	}

	cellData := make([][]string, 0, len(i.ExceptionCatches))
	for _, catch := range i.ExceptionCatches {
		if len(catch.ExceptionTypes) == 0 {
			// try finally only
			continue
		}

		cellData = append(cellData, []string{fmt.Sprintf("%s#%s", catch.ClassName, catch.MethodName), strings.Join(catch.ExceptionTypes, ", ")})
	}

	i.writeSectionHeader("%d Exception Catches", len(cellData))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Location", "Exception Types"}, cellData, true, -1)
	})

	return
}
