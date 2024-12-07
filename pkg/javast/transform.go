package javast

import (
	"encoding/json"
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
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
	return d.Invoke("replace-service", string(jsonArgs))
}

func InjectTransactionManager(component manifest.ComponentInfo) error {
	if component.Transform.TxManager == "" {
		return nil
	}

	ledger.Get().TransformTransaction(component.Name, component.Transform.TxManager)
	d := NewJavastDriver(component)
	return d.Invoke("inject-transaction-manager", component.Transform.TxManager)
}
