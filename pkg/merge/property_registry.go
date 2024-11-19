package merge

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"

	"federate/pkg/manifest"
)

func (cm *PropertyManager) registerProperty(component manifest.ComponentInfo, key string, value interface{}, filePath string) {
	if cm.resolvedProperties[component.Name] == nil {
		cm.resolvedProperties[component.Name] = make(map[string]PropertySource)
	}

	if cm.isReservedProperty(key) {
		// 保留字
		cm.registerReservedProperty(key, component, value)
	} else if val, overridden := cm.m.PropertyOverridden(key); overridden {
		// 用户手工指定值
		cm.resolvedProperties[component.Name][key] = PropertySource{
			Value:    val,
			FilePath: "overridden.yml", // yaml可以还原数据类型，而properties的值只能是string，因此这些key都放到yaml
		}
	} else {
		// 检查属性是否已存在
		if existingProp, exists := cm.resolvedProperties[component.Name][key]; exists {
			// 如果现有值不为空且新值是引用或为空，保留现有值
			if existingProp.Value != nil && (value == nil || (reflect.TypeOf(value).Kind() == reflect.String && strings.Contains(value.(string), "${"))) {
				if !cm.silent {
					log.Printf("[%s] Keeping existing value for %s: %v (new value was: %v)", component.Name, key, existingProp.Value, value)
				}
				return
			}
		}

		// 注册新值
		cm.resolvedProperties[component.Name][key] = PropertySource{
			Value:          value,
			OriginalString: fmt.Sprintf("%v", value),
			FilePath:       filePath,
		}
	}
}

func (cm *PropertyManager) resolveConflict(componentName string, key Key, value interface{}) {
	strKey := string(key)
	originalSource := cm.resolvedProperties[componentName][strKey]

	newOriginalString := originalSource.OriginalString
	if strings.Contains(newOriginalString, "${") {
		// 更新 OriginalString 中的引用 ${foo} => ${component1.foo}
		newOriginalString = cm.updateReferencesInString(originalSource.OriginalString, componentName)
		if !cm.silent {
			log.Printf("[%s] Key=%s Ref Updated: %s => %s", componentName, strKey, originalSource.OriginalString, newOriginalString)
		}
	}

	// Update the resolvedProperties with the prefixed key, for .properties && .yml
	nsKey := key.WithNamespace(componentName)
	cm.resolvedProperties[componentName][nsKey] = PropertySource{
		Value:          value,
		OriginalString: newOriginalString,
		FilePath:       originalSource.FilePath,
	}

	// 原有的key删除可能会造成间接依赖的jar里引用找不到：use property.override
	//delete(cm.resolvedProperties[componentName], strKey)
}

func (cm *PropertyManager) updateReferencesInString(s, componentName string) string {
	return os.Expand(s, func(key string) string {
		return "${" + componentName + "." + key + "}"
	})
}
