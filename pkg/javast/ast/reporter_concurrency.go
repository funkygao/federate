package ast

import (
	"fmt"

	"federate/pkg/tabular"
)

func (i *Info) showConcurrencyUsages() (empty bool) {
	if len(i.ConcurrencyUsages) == 0 {
		return true
	}

	data := make([][]string, 0, len(i.ConcurrencyUsages))
	for _, usage := range i.ConcurrencyUsages {
		data = append(data, []string{
			fmt.Sprintf("%s#%s", usage.ClassName, usage.MethodName),
			usage.Type,
			usage.Details,
		})
	}

	i.writeSectionHeader("%d Concurrency Usages", len(data))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Location", "Type", "Details"}, data, true, -1)
	})

	return false
}
