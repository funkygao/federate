package property

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDependencyGraph(t *testing.T) {
	t.Run("AddNode and AddDependency", func(t *testing.T) {
		dg := NewDependencyGraph()
		dg.AddNode("key1", "value1")
		dg.AddNode("key2", "value2")
		dg.AddDependency("key1", "key2")

		assert.Len(t, dg.Nodes, 2)
		assert.Equal(t, "value1", dg.Nodes["key1"].Value)
		assert.Equal(t, "value2", dg.Nodes["key2"].Value)
		assert.Contains(t, dg.Nodes["key1"].Deps, "key2")
	})

	t.Run("TopologicalSort", func(t *testing.T) {
		dg := NewDependencyGraph()
		dg.AddNode("key1", "value1")
		dg.AddNode("key2", "value2")
		dg.AddNode("key3", "value3")
		dg.AddDependency("key1", "key2")
		dg.AddDependency("key2", "key3")

		sorted := dg.TopologicalSort()
		assert.Equal(t, []string{"key3", "key2", "key1"}, sorted)
	})

	t.Run("TopologicalSort with cycle", func(t *testing.T) {
		dg := NewDependencyGraph()
		dg.AddNode("key1", "value1")
		dg.AddNode("key2", "value2")
		dg.AddNode("key3", "value3")
		dg.AddDependency("key1", "key2")
		dg.AddDependency("key2", "key3")
		dg.AddDependency("key3", "key1")

		sorted := dg.TopologicalSort()
		assert.Len(t, sorted, 3)
		assert.Contains(t, sorted, "key1")
		assert.Contains(t, sorted, "key2")
		assert.Contains(t, sorted, "key3")
	})

	t.Run("FromPropertyManager", func(t *testing.T) {
		pm := &PropertyManager{
			r: &registry{
				resolvableEntries: map[string]map[string]PropertyEntry{
					"component1": {
						"prop1": {Value: "value1"},
						"prop2": {Value: "${prop3}"},
						"prop5": {Value: "${component2.prop3}"},
					},
					"component2": {
						"prop3": {Value: "value3"},
						"prop4": {Value: "${component1.prop1}"},
					},
				},
			},
		}

		dg := NewDependencyGraph().FromPropertyManager(pm)

		assert.Len(t, dg.Nodes, 6)
		assert.Contains(t, dg.Nodes, "component1.prop1")
		assert.Contains(t, dg.Nodes, "component1.prop2")
		assert.Contains(t, dg.Nodes, "component1.prop3")
		assert.Contains(t, dg.Nodes, "component1.prop5")
		assert.Contains(t, dg.Nodes, "component2.prop3")
		assert.Contains(t, dg.Nodes, "component2.prop4")

		assert.Contains(t, dg.Nodes["component1.prop2"].Deps, "component1.prop3")
		assert.Contains(t, dg.Nodes["component1.prop5"].Deps, "component2.prop3")
		assert.Contains(t, dg.Nodes["component2.prop4"].Deps, "component1.prop1")

		sorted := dg.TopologicalSort()
		assert.Equal(t, 6, len(sorted))

		// Check the order of specific dependencies
		assert.True(t, indexOf(sorted, "component1.prop1") < indexOf(sorted, "component2.prop4"))
		assert.True(t, indexOf(sorted, "component2.prop3") < indexOf(sorted, "component1.prop5"))
		assert.True(t, indexOf(sorted, "component1.prop3") < indexOf(sorted, "component1.prop2"))

		// Print the sorted order for debugging
		t.Logf("Sorted order: %v", sorted)
	})

	t.Run("FromPropertyManager with nil", func(t *testing.T) {
		dg := NewDependencyGraph().FromPropertyManager(nil)
		assert.NotNil(t, dg)
		assert.Len(t, dg.Nodes, 0)
	})
}

func indexOf(slice []string, item string) int {
	for i, v := range slice {
		if v == item {
			return i
		}
	}
	return -1
}
