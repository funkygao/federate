package property

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"federate/pkg/manifest"
)

func (pm *PropertyManager) registerReservedProperty(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := pm.reservedValues[key]; !exists {
		pm.reservedValues[key] = []ComponentKeyValue{}
	}
	pm.reservedValues[key] = append(pm.reservedValues[key], ComponentKeyValue{Component: component, Value: value})
}

func (pm *PropertyManager) registerProperty(component manifest.ComponentInfo, key string, value interface{}, filePath string) {
	if pm.resolvedProperties[component.Name] == nil {
		pm.resolvedProperties[component.Name] = make(map[string]PropertySource)
	}

	// 保留字
	if pm.isReservedProperty(key) {
		pm.registerReservedProperty(key, component, value)
		return
	}

	// 用户手工指定值
	if val, overridden := pm.m.PropertyOverridden(key); overridden {
		pm.resolvedProperties[component.Name][key] = PropertySource{
			Value:    val,
			FilePath: fakeFile, // yaml可以还原数据类型，而properties的值只能是string，因此这些key都放到yaml
		}
		return
	}

	existingProp, exists := pm.getComponentProperty(component, key)
	if exists && pm.shouldKeepExistingValue(existingProp, value) {
		if !pm.silent {
			log.Printf("[%s] Keep existing value for %s: %v (new value was: %v)", component.Name, key, existingProp.Value, value)
		}
		return
	}

	// 注册新值
	pm.resolvedProperties[component.Name][key] = PropertySource{
		Value:          value,
		OriginalString: fmt.Sprintf("%v", value),
		FilePath:       filePath,
	}
}

func (pm *PropertyManager) getComponentProperty(component manifest.ComponentInfo, key string) (*PropertySource, bool) {
	if pm.resolvedProperties[component.Name] == nil {
		return nil, false
	}

	existingProp, exists := pm.resolvedProperties[component.Name][key]
	return &existingProp, exists
}

func (pm *PropertyManager) shouldKeepExistingValue(existing *PropertySource, newValue interface{}) bool {
	return existing.Value != nil && (newValue == nil || (reflect.TypeOf(newValue).Kind() == reflect.String && strings.Contains(newValue.(string), "${")))
}
