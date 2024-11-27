package property

import (
	"fmt"
	"log"
	"os"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/merge/transformer"
)

type registry struct {
	manifest *manifest.Manifest
	pm       *PropertyManager

	silent bool

	resolvableEntries   map[string]map[string]PropertyEntry
	unresolvableEntries map[string]map[string]PropertyEntry

	reservedPropertyValues map[string][]ComponentPropertyValue
}

func newRegistry(m *manifest.Manifest, silent bool) *registry {
	return &registry{
		resolvableEntries:      make(map[string]map[string]PropertyEntry),
		unresolvableEntries:    make(map[string]map[string]PropertyEntry),
		reservedPropertyValues: make(map[string][]ComponentPropertyValue),
		manifest:               m,
		silent:                 silent,
	}
}

// AddProperty adds a property to the registry
func (r *registry) AddProperty(component manifest.ComponentInfo, key string, value interface{}, filePath string) {
	if r.resolvableEntries[component.Name] == nil {
		r.resolvableEntries[component.Name] = make(map[string]PropertyEntry)
	}

	// 保留关键字
	if r.isReservedProperty(key) {
		r.addReservedProperty(key, component, value)
		return
	}

	// 用户自定义
	if val, overridden := r.manifest.PropertyOverridden(key); overridden {
		r.resolvableEntries[component.Name][key] = PropertyEntry{
			Value:    val,
			FilePath: fakeFile,
		}
		return
	}

	existingEntry, exists := r.getComponentProperty(component, key)
	if exists && r.shouldKeepExistingValue(existingEntry, value) {
		if !r.silent {
			log.Printf("[%s] Keep existing value for %s: %v (new value was: %v)", component.Name, key, existingEntry.Value, value)
		}
		return
	}

	r.resolvableEntries[component.Name][key] = PropertyEntry{
		Value:     value,
		RawString: fmt.Sprintf("%v", value),
		FilePath:  filePath,
	}
}

// UpdateProperty updates an existing property in the registry
func (r *registry) UpdateProperty(componentName string, key string, existingEntry PropertyEntry, newValue interface{}) {
	r.resolvableEntries[componentName][key] = PropertyEntry{
		Value:     newValue,
		RawString: existingEntry.RawString,
		FilePath:  existingEntry.FilePath,
	}
}

// MarkAsUnresolvable marks a property as unresolvable and removes it from resolvable entries
func (r *registry) MarkAsUnresolvable(componentName string, existingEntry PropertyEntry, key string) {
	if _, present := r.unresolvableEntries[componentName]; !present {
		r.unresolvableEntries[componentName] = make(map[string]PropertyEntry)
	}

	r.unresolvableEntries[componentName][key] = existingEntry
	delete(r.resolvableEntries[componentName], key)
}

// GetAllResolvableEntries returns all resolvable entries
func (r *registry) GetAllResolvableEntries() map[string]map[string]PropertyEntry {
	return r.resolvableEntries
}

// getComponentProperty retrieves a property entry for a component
func (r *registry) getComponentProperty(component manifest.ComponentInfo, key string) (*PropertyEntry, bool) {
	if r.resolvableEntries[component.Name] == nil {
		return nil, false
	}

	existingEntry, exists := r.resolvableEntries[component.Name][key]
	return &existingEntry, exists
}

func (r *registry) ComponentYamlEntries(component manifest.ComponentInfo) map[string]PropertyEntry {
	result := make(map[string]PropertyEntry)
	for key, entry := range r.resolvableEntries[component.Name] {
		if entry.IsYAML() {
			result[key] = entry
		}
	}
	return result
}

func (r *registry) ComponentPropertiesEntries(component manifest.ComponentInfo) map[string]PropertyEntry {
	result := make(map[string]PropertyEntry)
	for key, entry := range r.resolvableEntries[component.Name] {
		if entry.IsProperties() {
			result[key] = entry
		}
	}
	return result
}

// GetReservedPropertyValues returns the reserved property values
func (r *registry) GetReservedPropertyValues() map[string][]ComponentPropertyValue {
	return r.reservedPropertyValues
}

func (r *registry) ResolvePropertyReference(component, value string) interface{} {
	return os.Expand(value, func(key string) string {
		// 首先在当前组件中查找，包括 YAML 和 Properties 文件
		if entry, ok := r.resolvableEntries[component][key]; ok {
			return fmt.Sprintf("%v", entry.Value)
		}

		// 如果在当前组件中找不到，则在所有其他组件中查找
		for thatComponent, entries := range r.resolvableEntries {
			if thatComponent != component {
				if entry, ok := entries[key]; ok {
					return fmt.Sprintf("%v", entry.Value)
				}
			}
		}

		// 如果找不到引用的值，返回原始占位符
		return "${" + key + "}"
	})
}

// shouldKeepExistingValue 决定是否应保留现有属性值而不是用新值替换它
func (r *registry) shouldKeepExistingValue(existing *PropertyEntry, newValue interface{}) bool {
	// 如果现有值为空，允许新值替换
	if existing.Value == nil {
		return false
	}

	// 如果新值为空，保留现有值
	if newValue == nil {
		return true
	}

	// 如果新值是字符串并且包含占位符（如 ${...}），保留现有值
	// 这是为了避免用未解析的引用替换可能已经解析过的值
	if str, ok := newValue.(string); ok {
		return strings.Contains(str, "${")
	}

	// 其他情况下，允许新值替换现有值
	return false
}

// 调和冲突
func (r *registry) NamespaceProperty(componentName string, key Key, value interface{}) {
	strKey := string(key)
	originalEntry := r.resolvableEntries[componentName][strKey]

	newRawString := originalEntry.RawString
	if strings.Contains(newRawString, "${") {
		newRawString = r.pm.namespacePropertyPlaceholders(originalEntry.RawString, componentName)
		if !r.silent {
			log.Printf("[%s] Key=%s Ref Updated: %s => %s", componentName, strKey, originalEntry.RawString, newRawString)
		}
	}

	if integralKey := r.getConfigurationPropertiesPrefix(strKey); integralKey != "" {
		r.namespaceConfigurationProperties(componentName, integralKey)
	} else {
		r.namespaceRegularProperty(componentName, key, value, newRawString, originalEntry)
	}
}

// 统一调和 @ConfigurationProperties 的 key
func (r *registry) namespaceConfigurationProperties(componentName, configPropPrefix string) {
	for subKey := range r.resolvableEntries[componentName] {
		if strings.HasPrefix(subKey, configPropPrefix) {
			transformer.Get().TransformConfigurationProperties(componentName, configPropPrefix, Key(configPropPrefix).WithNamespace(componentName))
			nsKey := Key(subKey).WithNamespace(componentName)
			// 为该key增加ns前缀的key
			r.resolvableEntries[componentName][nsKey] = PropertyEntry{
				Value:     r.resolvableEntries[componentName][subKey].Value,
				RawString: r.resolvableEntries[componentName][subKey].RawString,
				FilePath:  r.resolvableEntries[componentName][subKey].FilePath,
			}
		}
	}
}

func (r *registry) namespaceRegularProperty(componentName string, key Key, value interface{}, newOriginalString string, originalEntry PropertyEntry) {
	nsKey := key.WithNamespace(componentName)
	transformer.Get().TransformRegularProperty(componentName, string(key), nsKey)
	r.resolvableEntries[componentName][nsKey] = PropertyEntry{
		Value:     value,
		RawString: newOriginalString,
		FilePath:  originalEntry.FilePath,
	}
}

func (r *registry) isReservedProperty(key string) bool {
	_, exists := reservedKeyHandlers[key]
	return exists
}

func (r *registry) addReservedProperty(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := r.reservedPropertyValues[key]; !exists {
		r.reservedPropertyValues[key] = []ComponentPropertyValue{}
	}
	r.reservedPropertyValues[key] = append(r.reservedPropertyValues[key], ComponentPropertyValue{Component: component, Value: value})
}

func (r *registry) getConfigurationPropertiesPrefix(key string) string {
	for _, prefix := range r.manifest.Main.Reconcile.Resources.Property.ConfigurationPropertiesKeys {
		if strings.HasPrefix(key, prefix) {
			return prefix
		}
	}
	return ""
}
