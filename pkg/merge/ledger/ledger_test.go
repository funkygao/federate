package ledger

import (
	"testing"
)

func TestTransformer(t *testing.T) {
	// Test TransformRequestMapping
	instance.TransformRequestMapping("component1", "/old-path", "/new-path")
	if instance.requestMappings["component1"]["/old-path"] != "/new-path" {
		t.Errorf("TransformRequestMapping failed")
	}

	// Test TransformRegularProperty
	instance.TransformRegularProperty("component2", "oldProp", "newProp")
	if instance.regularProperties["component2"]["oldProp"] != "newProp" {
		t.Errorf("TransformRegularProperty failed")
	}

	// Test TransformConfigurationProperties
	instance.TransformConfigurationProperties("component3", "oldConfig", "newConfig")
	if instance.configurationProperties["component3"]["oldConfig"] != "newConfig" {
		t.Errorf("TransformConfigurationProperties failed")
	}

	// Test TransformImportResource
	instance.TransformImportResource("component4", "oldResource", "newResource")
	if instance.importResource["component4"]["oldResource"] != "newResource" {
		t.Errorf("TransformImportResource failed")
	}

	// Test TransformTransaction
	instance.TransformTransaction("component5", "txManager")
	if instance.transactions["component5"] != "txManager" {
		t.Errorf("TransformTransaction failed")
	}

	// Test RegisterEnvKey
	instance.RegisterEnvKey("component6", "ENV_KEY")
	if _, exists := instance.envKeys["component6"]["ENV_KEY"]; !exists {
		t.Errorf("RegisterEnvKey failed")
	}

	// Test ShowSummary (just ensure it doesn't panic)
	instance.ShowSummary()
}
