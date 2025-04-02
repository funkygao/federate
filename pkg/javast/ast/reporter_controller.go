package ast

import (
	"federate/pkg/tabular"
)

func (i *Info) showRestControllersReport() (empty bool) {
	if len(i.RestControllers) == 0 {
		return true
	}

	i.writeSectionHeader("REST Controllers and Endpoints")

	var cellData [][]string
	for _, controller := range i.RestControllers {
		i.writeSectionBody(func() {
			for _, endpoint := range controller.Endpoints {
				cellData = append(cellData, []string{
					controller.ClassName,
					controller.BasePath,
					endpoint.MethodName,
					endpoint.Path,
				})
			}
		})
	}
	tabular.Display([]string{"Controller", "BasePath", "Method", "Path"}, cellData, true, -1)

	return false
}
