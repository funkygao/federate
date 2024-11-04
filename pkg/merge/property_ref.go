package merge

import (
	"fmt"
	"log"
	"os"
)

func (cm *PropertyManager) resolveAllReferences() {
	resolved := make(map[string]bool)
	for len(cm.propertyReferences) > 0 {
		unresolved := make([]PropertyReference, 0)
		for _, ref := range cm.propertyReferences {
			newValue := cm.resolvePropertyReference(ref.Component, ref.Value)
			if newValue != ref.Value {
				cm.resolvedProperties[ref.Component][ref.Key] = newValue
				resolved[ref.Component+"."+ref.Key] = true
				log.Printf("[%s] %s: %s => %s", ref.Component, ref.Key, ref.Value, newValue)
			} else {
				unresolved = append(unresolved, ref)
			}
		}
		if len(unresolved) == len(cm.propertyReferences) {
			// 如果没有解析任何引用，退出循环以避免无限循环
			break
		}
		cm.propertyReferences = unresolved
	}

	// 处理任何剩余的未解析引用
	for _, ref := range cm.propertyReferences {
		if !resolved[ref.Component+"."+ref.Key] {
			log.Printf("Warning: Unresolved reference in [%s] %s: %s", ref.Component, ref.Key, ref.Value)
		}
	}
}

func (cm *PropertyManager) resolvePropertyReference(component, value string) string {
	return os.Expand(value, func(key string) string {
		// 首先在当前组件中查找
		if v, ok := cm.resolvedProperties[component][key]; ok {
			return fmt.Sprintf("%v", v)
		}
		// 如果在当前组件中找不到，则在所有组件中查找
		for _, props := range cm.resolvedProperties {
			if v, ok := props[key]; ok {
				return fmt.Sprintf("%v", v)
			}
		}
		// 如果找不到引用的值，返回原始占位符
		return "${" + key + "}"
	})
}
