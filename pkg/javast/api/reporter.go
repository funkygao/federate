package api

import (
	"sort"
	"strings"

	"federate/pkg/prompt"
	"federate/pkg/tabular"
)

func (i *Info) ShowReport() {
	var logger *prompt.PromptLogger
	if GeneratePrompt {
		logger = prompt.NewPromptLogger()
		logger.AddPrompt(prompt.JavaAPI)
		logger.Start()
		defer logger.Stop()
	}

	var cellData [][]string

	// 获取所有接口名称并排序
	interfaceNames := make([]string, 0, len(*i))
	for name := range *i {
		interfaceNames = append(interfaceNames, name)
	}
	sort.Strings(interfaceNames)

	for _, interfaceName := range interfaceNames {
		interfaceInfo := (*i)[interfaceName]

		for _, method := range interfaceInfo.Methods {
			params := make([]string, len(method.Parameters))
			for j, param := range method.Parameters {
				params[j] = param.Type + " " + param.Name
			}
			paramStr := strings.Join(params, ", ")

			cellData = append(cellData, []string{interfaceName, method.Name, paramStr, method.ReturnType})
		}
	}

	i.writeSectionBody(logger, func() {
		tabular.Display([]string{"Interface", "Method", "Parameters", "Return Type"}, cellData, true, -1)
	})
}

func (i *Info) writeSectionBody(logger *prompt.PromptLogger, content func()) {
	if GeneratePrompt {
		logger.AddPrompt("```\n")
		content()
		logger.AddPrompt("```\n\n")
	} else {
		content()
	}
}
