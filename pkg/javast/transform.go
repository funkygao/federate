package javast

import (
	"encoding/json"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
)

const (
	CmdReplaceService       = "replace-service"
	CmdInjectTrxManager     = "inject-transaction-manager"
	CmdUpdatePropertyRefKey = "update-property-keys"
)

var cmds []Command

func Instrument(m *manifest.Manifest) error {
	for _, c := range m.Components {
		driver := NewJavastDriver(c)
		if err := driver.Invoke(cmds...); err != nil {
			return err
		}
	}

	return nil
}

func TransformComponentBean(component manifest.ComponentInfo) {
	if len(component.Transform.Services) == 0 {
		return
	}

	jsonArgs, _ := json.Marshal(component.Transform.Services)
	cmds = append(cmds, Command{CmdReplaceService, string(jsonArgs)})
}

func InjectTransactionManager(component manifest.ComponentInfo) {
	if component.Transform.TxManager == "" {
		return
	}

	ledger.Get().TransformTransaction(component.Name, component.Transform.TxManager)
	cmds = append(cmds, Command{CmdInjectTrxManager, component.Transform.TxManager})
}

func UpdatePropertyKeys(component manifest.ComponentInfo, keyMapping map[string]string) {
	if len(keyMapping) == 0 {
		return
	}

	jsonArgs, _ := json.Marshal(keyMapping)
	cmds = append(cmds, Command{CmdUpdatePropertyRefKey, string(jsonArgs)})
}
