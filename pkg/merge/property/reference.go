package property

import (
	"fmt"
	"log"
	"strings"

	"federate/pkg/tabular"
)

func (pm *PropertyManager) resolveAllReferences() {
	dg := NewDependencyGraph().FromPropertyManager(pm)
	sortedKeys := dg.TopologicalSort()

	for _, fullKey := range sortedKeys {
		component, key := splitFullKey(fullKey)
		if entry, ok := pm.r.resolvableEntries[component][key]; ok {
			if rawRef := entry.RawReferenceValue(); rawRef != "" {
				newValue := pm.r.ResolvePropertyReference(component, rawRef)
				if newValue != rawRef {
					pm.r.ResolveProperty(component, key, entry, newValue)
				}
			}
		}
	}

	// 第二遍解析，处理可能依赖于刚刚解析的值的引用
	for _, fullKey := range sortedKeys {
		component, key := splitFullKey(fullKey)
		if entries, ok := pm.r.resolvableEntries[component]; ok {
			if entry, ok := entries[key]; ok {
				if rawRef := entry.RawReferenceValue(); rawRef != "" {
					newValue := pm.r.ResolvePropertyReference(component, rawRef)
					if newValue != rawRef {
						pm.r.ResolveProperty(component, key, entry, newValue)
					}
				}
			}
		}
	}

	// 删除所有未解析的引用并输出警告
	var unresolved [][]string
	for component, entries := range pm.r.GetAllResolvableEntries() {
		for key, existingEntry := range entries {
			if strValue := existingEntry.StringValue(); strValue != "" {
				if rawRef := existingEntry.RawReferenceValue(); rawRef != "" {
					pm.r.MovePropertyToUnresolvable(component, existingEntry, key)
					unresolved = append(unresolved, []string{component, key, strValue, fmt.Sprintf("%v", existingEntry.IsYAML())})
				} else {
					pm.r.ResolveProperty(component, key, existingEntry, strValue)
				}
			}
		}
	}

	if !pm.silent && len(unresolved) > 0 {
		header := []string{"Component", "Key", "Unresolved Value", "Yaml"}
		log.Printf("Found %d unresolvable references (these properties will be removed):", len(unresolved))
		tabular.Display(header, unresolved, false, -1)
	}
}

func splitFullKey(fullKey string) (component, key string) {
	parts := strings.SplitN(fullKey, ".", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "", fullKey
}
