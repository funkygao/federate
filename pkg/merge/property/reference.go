package property

import (
	"fmt"
	"log"

	"federate/pkg/tablerender"
)

func (cm *PropertyManager) resolveAllReferences() {
	maxIterations := 10 // 设置最大迭代次数，防止无限循环
	iteration := 0
	changed := true

	for changed && iteration < maxIterations {
		changed = false
		iteration++

		for component, entries := range cm.r.GetAllResolvableEntries() {
			for key, existingEntry := range entries {
				if rawRef := existingEntry.RawReferenceValue(); rawRef != "" {
					newValue := cm.r.ResolvePropertyReference(component, rawRef)
					if newValue != rawRef {
						cm.r.UpdateProperty(component, key, existingEntry, newValue)
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
	for component, entries := range cm.r.GetAllResolvableEntries() {
		for key, existingEntry := range entries {
			if strValue := existingEntry.StringValue(); strValue != "" {
				if rawRef := existingEntry.RawReferenceValue(); rawRef != "" {
					cm.r.MarkAsUnresolvable(component, existingEntry, key)
					unresolved = append(unresolved, []string{component, key, strValue, fmt.Sprintf("%v", existingEntry.IsYAML())})
				} else {
					cm.r.UpdateProperty(component, key, existingEntry, strValue)
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
