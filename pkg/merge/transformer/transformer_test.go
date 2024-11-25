package transformer

import (
	"testing"
)

func TestTransformer(t *testing.T) {
	// Test TransformRequestMapping
	tf.TransformRequestMapping("component1", "/old-path", "/new-path")
	if tf.requestMappings["component1"]["/old-path"] != "/new-path" {
		t.Errorf("TransformRequestMapping failed")
	}

	// Test TransformRegularProperty
	tf.TransformRegularProperty("component2", "oldProp", "newProp")
	if tf.regularProperties["component2"]["oldProp"] != "newProp" {
		t.Errorf("TransformRegularProperty failed")
	}

	// Test TransformConfigurationProperties
	tf.TransformConfigurationProperties("component3", "oldConfig", "newConfig")
	if tf.configurationProperties["component3"]["oldConfig"] != "newConfig" {
		t.Errorf("TransformConfigurationProperties failed")
	}

	// Test TransformImportResource
	tf.TransformImportResource("component4", "oldResource", "newResource")
	if tf.importResource["component4"]["oldResource"] != "newResource" {
		t.Errorf("TransformImportResource failed")
	}

	// Test TransformTransaction
	tf.TransformTransaction("component5", "txManager")
	if tf.transactions["component5"] != "txManager" {
		t.Errorf("TransformTransaction failed")
	}

	// Test RegisterEnvKey
	tf.RegisterEnvKey("component6", "ENV_KEY")
	if _, exists := tf.envKeys["component6"]["ENV_KEY"]; !exists {
		t.Errorf("RegisterEnvKey failed")
	}

	// Test ShowSummary (just ensure it doesn't panic)
	tf.ShowSummary()
}
