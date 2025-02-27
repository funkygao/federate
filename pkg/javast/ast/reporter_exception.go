package ast

import (
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showExceptionCatches() (empty bool) {
	if !ShowException || len(i.ExceptionCatches) == 0 {
		return true
	}

	cellData := make([][]string, 0, len(i.ExceptionCatches))
	for _, catch := range i.ExceptionCatches {
		if len(catch.ExceptionTypes) == 0 {
			// try finally only
			continue
		}

		cellData = append(cellData, []string{catch.ClassName, catch.MethodName, strings.Join(catch.ExceptionTypes, ", ")})
	}

	i.writeSectionHeader("%d Exception Catches", len(cellData))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Class", "Method", "Catch Exception Types"}, cellData, true, -1)
	})

	return
}

func (i *Info) showMethodThrows() (empty bool) {
	if !ShowException || len(i.MethodThrows) == 0 {
		return true
	}

	data := make([][]string, 0, len(i.MethodThrows))
	for _, info := range i.MethodThrows {
		data = append(data, []string{info.ClassName, info.MethodName, strings.Join(info.ThrownExceptions, ", ")})
	}

	i.writeSectionHeader("%d Methods with Throws Declarations", len(data))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Class", "Method", "Thrown Exceptions"}, data, false, -1)
	})

	return false
}
