package property

import (
	"fmt"
	"log"
	"os"

	"federate/pkg/tablerender"
)

func (cm *PropertyManager) resolveAllReferences() {
	maxIterations := 10 // 设置最大迭代次数，防止无限循环
	iteration := 0
	changed := true

	for changed && iteration < maxIterations {
		changed = false
		iteration++

		for component, props := range cm.resolvableEntries {
			for key, existingEntry := range props {
				if rawRef := existingEntry.RawReferenceValue(); rawRef != "" {
					newValue := cm.resolvePropertyReference(component, rawRef)
					if newValue != rawRef {
						cm.updatePropertyEntry(component, key, existingEntry, newValue)
						changed = true
					} else {
						// 相互引用，此轮还无法解析
					}
				}
			}
		}
	}

	// 删除所有未解析的引用并输出警告
	var unresolved [][]string
	for component, props := range cm.resolvableEntries {
		for key, existingEntry := range props {
			if strValue := existingEntry.StringValue(); strValue != "" {
				if rawRef := existingEntry.RawReferenceValue(); rawRef != "" {
					cm.registerUnsolvableProperty(component, existingEntry, key)
					unresolved = append(unresolved, []string{component, key, strValue, fmt.Sprintf("%v", existingEntry.IsYAML())})
				} else {
					cm.updatePropertyEntry(component, key, existingEntry, strValue)
				}
			}
		}
	}

	if !cm.silent && len(unresolved) > 0 {
		header := []string{"Component", "Key", "Unresolved Value", "Yaml"}
		log.Printf("Found %d unresolved references (these properties will be removed) after %d iterations:", len(unresolved), iteration+1)
		tablerender.DisplayTable(header, unresolved, false, -1)
	}
}

func (cm *PropertyManager) resolvePropertyReference(component, value string) interface{} {
	return os.Expand(value, func(key string) string {
		// 首先在当前组件中查找，包括 YAML 和 Properties 文件
		if propSource, ok := cm.resolvableEntries[component][key]; ok {
			return fmt.Sprintf("%v", propSource.Value) // TODO
		}

		// 如果在当前组件中找不到，则在所有其他组件中查找
		for otherComponent, props := range cm.resolvableEntries {
			if otherComponent != component {
				if propSource, ok := props[key]; ok {
					return fmt.Sprintf("%v", propSource.Value)
				}
			}
		}

		// 如果找不到引用的值，返回原始占位符
		return "${" + key + "}"
	})
}
