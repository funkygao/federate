package merge

import (
	"fmt"
	"log"
	"os"
	"strings"

	"federate/pkg/tablerender"
)

func (cm *PropertyManager) resolveAllReferences() {
	maxIterations := 10 // 设置最大迭代次数，防止无限循环
	iteration := 0
	changed := true

	for changed && iteration < maxIterations {
		changed = false
		iteration++

		for component, props := range cm.resolvedProperties {
			for key, propSource := range props {
				if strValue, ok := propSource.Value.(string); ok && strings.Contains(strValue, "${") {
					newValue := cm.resolvePropertyReference(component, strValue)
					if newValue != strValue {
						cm.resolvedProperties[component][key] = PropertySource{
							Value:          newValue,
							OriginalString: propSource.OriginalString,
							FilePath:       propSource.FilePath,
						}
						changed = true
					}
				}
			}
		}
	}

	// 删除所有未解析的引用并输出警告
	var unresolved [][]string
	for component, props := range cm.resolvedProperties {
		for key, propSource := range props {
			if strValue, ok := propSource.Value.(string); ok {
				if strings.Contains(strValue, "${") {
					if _, present := cm.unresolvedProperties[component]; !present {
						cm.unresolvedProperties[component] = make(map[string]PropertySource)
					}
					unresolved = append(unresolved, []string{component, key, strValue, fmt.Sprintf("%v", propSource.IsYAML())})
					cm.unresolvedProperties[component][key] = propSource
					delete(cm.resolvedProperties[component], key)
				} else {
					// 确保值不包含 ${}
					cm.resolvedProperties[component][key] = PropertySource{
						Value:          strings.ReplaceAll(strValue, "${", ""),
						OriginalString: propSource.OriginalString,
						FilePath:       propSource.FilePath,
					}
				}
			}
		}
	}

	if len(unresolved) > 0 {
		header := []string{"Component", "Key", "Unresolved Value", "Yaml"}
		log.Printf("Found %d unresolved references (these properties will be removed) after %d iterations:", len(unresolved), iteration+1)
		tablerender.DisplayTable(header, unresolved, false, -1)
	}
}

func (cm *PropertyManager) resolvePropertyReference(component, value string) interface{} {
	return os.Expand(value, func(key string) string {
		// 首先在当前组件中查找，包括 YAML 和 Properties 文件
		if propSource, ok := cm.resolvedProperties[component][key]; ok {
			return fmt.Sprintf("%v", propSource.Value) // TODO
		}

		// 如果在当前组件中找不到，则在所有其他组件中查找
		for otherComponent, props := range cm.resolvedProperties {
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
