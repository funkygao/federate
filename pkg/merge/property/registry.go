package property

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"federate/pkg/manifest"
)

func (pm *PropertyManager) registerReservedProperty(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := pm.reservedProperties[key]; !exists {
		pm.reservedProperties[key] = []ComponentKeyValue{}
	}
	pm.reservedProperties[key] = append(pm.reservedProperties[key], ComponentKeyValue{Component: component, Value: value})
}

func (pm *PropertyManager) registerNewProperty(component manifest.ComponentInfo, key string, value interface{}, filePath string) {
	if pm.resolvableEntries[component.Name] == nil {
		pm.resolvableEntries[component.Name] = make(map[string]PropertyEntry)
	}

	// 保留字
	if pm.isReservedProperty(key) {
		pm.registerReservedProperty(key, component, value)
		return
	}

	// 用户手工指定值
	if val, overridden := pm.m.PropertyOverridden(key); overridden {
		pm.resolvableEntries[component.Name][key] = PropertyEntry{
			Value:    val,
			FilePath: fakeFile, // yaml可以还原数据类型，而properties的值只能是string，因此这些key都放到yaml
		}
		return
	}

	existingEntry, exists := pm.getComponentProperty(component, key)
	if exists && pm.shouldKeepExistingValue(existingEntry, value) {
		if !pm.silent {
			log.Printf("[%s] Keep existing value for %s: %v (new value was: %v)", component.Name, key, existingEntry.Value, value)
		}
		return
	}

	// 注册新值
	pm.resolvableEntries[component.Name][key] = PropertyEntry{
		Value:     value,
		RawString: fmt.Sprintf("%v", value),
		FilePath:  filePath,
	}
}

// newValue 是解析后的值
func (pm *PropertyManager) updatePropertyEntry(componentName string, key string, existingEntry PropertyEntry, newValue interface{}) {
	pm.resolvableEntries[componentName][key] = PropertyEntry{
		Value:     newValue,
		RawString: existingEntry.RawString,
		FilePath:  existingEntry.FilePath,
	}
}

func (pm *PropertyManager) registerUnsolvableProperty(componentName string, existingEntry PropertyEntry, key string) {
	if _, present := pm.unresolvableEntries[componentName]; !present {
		pm.unresolvableEntries[componentName] = make(map[string]PropertyEntry)
	}

	pm.unresolvableEntries[componentName][key] = existingEntry

	// 从可解析里删除
	delete(pm.resolvableEntries[componentName], key)
}

func (pm *PropertyManager) getComponentProperty(component manifest.ComponentInfo, key string) (*PropertyEntry, bool) {
	if pm.resolvableEntries[component.Name] == nil {
		return nil, false
	}

	existingEntry, exists := pm.resolvableEntries[component.Name][key]
	return &existingEntry, exists
}

func (pm *PropertyManager) shouldKeepExistingValue(existing *PropertyEntry, newValue interface{}) bool {
	return existing.Value != nil && (newValue == nil || (reflect.TypeOf(newValue).Kind() == reflect.String && strings.Contains(newValue.(string), "${")))
}

func (pm *PropertyManager) updateResolvedProperties(componentName string, key Key, value interface{}) {
	strKey := string(key)
	originalSource := pm.resolvableEntries[componentName][strKey]

	// 已经占位符替换的，要恢复占位符
	newOriginalString := originalSource.RawString
	if strings.Contains(newOriginalString, "${") {
		// 更新 OriginalString 中的引用 ${foo} => ${component1.foo}
		newOriginalString = pm.updateReferencesInString(originalSource.RawString, componentName)
		if !pm.silent {
			log.Printf("[%s] Key=%s Ref Updated: %s => %s", componentName, strKey, originalSource.RawString, newOriginalString)
		}
	}

	// 修改 resolvedProperties，写盘使用
	if integralKey := pm.getConfigurationPropertiesPrefix(strKey); integralKey != "" {
		pm.handleConfigurationProperties(componentName, integralKey)
	} else {
		pm.handleRegularProperty(componentName, key, value, newOriginalString, originalSource)
	}
}

func (pm *PropertyManager) handleConfigurationProperties(componentName, configPropPrefix string) {
	// key 是 ConfigurationProperties 的一部分，为所有相关的 key 添加命名空间
	for subKey := range pm.resolvableEntries[componentName] {
		if strings.HasPrefix(subKey, configPropPrefix) {
			nsKey := Key(subKey).WithNamespace(componentName)
			pm.resolvableEntries[componentName][nsKey] = PropertyEntry{
				Value:     pm.resolvableEntries[componentName][subKey].Value,
				RawString: pm.resolvableEntries[componentName][subKey].RawString,
				FilePath:  pm.resolvableEntries[componentName][subKey].FilePath,
			}
		}
	}
}

func (pm *PropertyManager) handleRegularProperty(componentName string, key Key, value interface{}, newOriginalString string, originalEntry PropertyEntry) {
	nsKey := key.WithNamespace(componentName)
	pm.resolvableEntries[componentName][nsKey] = PropertyEntry{
		Value:     value,
		RawString: newOriginalString,
		FilePath:  originalEntry.FilePath,
	}
}
