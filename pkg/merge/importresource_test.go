package merge

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestProcessImportResource(t *testing.T) {
	testCases := []struct {
		input          string
		componentName  string
		expectedOutput string
	}{
		{
			`@ImportResource("classpath:strategy-config.xml")`,
			"myComponent",
			`@ImportResource("classpath:federated/myComponent/strategy-config.xml")`,
		},
		{
			`@ImportResource("strategy-config.xml")`,
			"myComponent",
			`@ImportResource("federated/myComponent/strategy-config.xml")`,
		},
		{
			`@ImportResource(locations = {"strategy-config.xml", "a/b/c.xml"})`,
			"myComponent",
			`@ImportResource(locations = {"federated/myComponent/strategy-config.xml", "federated/myComponent/a/b/c.xml"})`,
		},
		{
			`@ImportResource(locations = {"classpath:strategy-config.xml", "classpath:a/b/c.xml"})`,
			"myComponent",
			`@ImportResource(locations = {"classpath:federated/myComponent/strategy-config.xml", "classpath:federated/myComponent/a/b/c.xml"})`,
		},
		{
			`@ImportResource(locations = {"strategy-config.xml", "classpath:a/b/c.xml"})`,
			"myComponent",
			`@ImportResource(locations = {"federated/myComponent/strategy-config.xml", "classpath:federated/myComponent/a/b/c.xml"})`,
		},
		{
			`@ImportResource(locations = {"classpath:strategy-config.xml"})`,
			"myComponent",
			`@ImportResource("classpath:federated/myComponent/strategy-config.xml")`,
		},
		{
			`@ImportResource('classpath:strategy-config.xml')`,
			"myComponent",
			`@ImportResource("classpath:federated/myComponent/strategy-config.xml")`,
		},
	}

	manager := NewImportResourceManager(nil)
	for _, tc := range testCases {
		output, changed := manager.processImportResource(tc.input, tc.componentName)
		assert.True(t, changed, "Expected the input to be changed")
		assert.Equal(t, tc.expectedOutput, output, "The processed output does not match the expected output")
	}
}
