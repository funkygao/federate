package javast

import (
	"encoding/json"
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
)

const (
	CmdReplaceService       = "replace-service"
	CmdInjectTrxManager     = "inject-transaction-manager"
	CmdUpdatePropertyRefKey = "update-property-keys"
)

func TransformService(component manifest.ComponentInfo) error {
	if len(component.Transform.Services) == 0 {
		log.Printf("[%s] Skipped empty transform.service", component.Name)
		return nil
	}

	jsonArgs, err := json.Marshal(component.Transform.Services)
	if err != nil {
		return err
	}

	d := NewJavastDriver(component)
	return d.Invoke(Command{CmdReplaceService, string(jsonArgs)})
}

func InjectTransactionManager(component manifest.ComponentInfo) error {
	if component.Transform.TxManager == "" {
		return nil
	}

	ledger.Get().TransformTransaction(component.Name, component.Transform.TxManager)
	d := NewJavastDriver(component)
	return d.Invoke(Command{CmdInjectTrxManager, component.Transform.TxManager})
}

func UpdatePropertyKeys(component manifest.ComponentInfo, keyMapping map[string]string) error {
	if len(keyMapping) == 0 {
		log.Printf("[%s] No keys to update", component.Name)
		return nil
	}

	jsonArgs, err := json.Marshal(keyMapping)
	if err != nil {
		return err
	}

	d := NewJavastDriver(component)
	return d.Invoke(Command{CmdUpdatePropertyRefKey, string(jsonArgs)})
}
