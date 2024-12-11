package javast

import (
	"encoding/json"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
)

var backlog []Command

func Instrument() error {
	driver := NewJavastDriver()
	return driver.Invoke(backlog...)
}

func TransformResourceInject(component manifest.ComponentInfo) {
	backlog = append(backlog, Command{CmdTransformResourceInject, component.Name, component.Name})
}

func TransformImportResource(component manifest.ComponentInfo) {
	backlog = append(backlog, Command{CmdTransformImportResource, component.Name, component.Name})
}

func TransformComponentBean(component manifest.ComponentInfo) {
	if len(component.Transform.Services) == 0 {
		return
	}

	jsonArgs, _ := json.Marshal(component.Transform.Services)
	backlog = append(backlog, Command{CmdReplaceService, component.Name, string(jsonArgs)})
}

func InjectTransactionManager(component manifest.ComponentInfo) {
	if component.Transform.TxManager == "" {
		return
	}

	ledger.Get().TransformTransaction(component.Name, component.Transform.TxManager)
	backlog = append(backlog, Command{CmdInjectTrxManager, component.Name, component.Transform.TxManager})
}

func UpdatePropertyKeys(component manifest.ComponentInfo, keyMapping map[string]string) {
	if len(keyMapping) == 0 {
		return
	}

	jsonArgs, _ := json.Marshal(keyMapping)
	backlog = append(backlog, Command{CmdUpdatePropertyRefKey, component.Name, string(jsonArgs)})
}
