package property

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/merge/conflict"
)

// 合并 YAML, properties 文件
type PropertyManager struct {
	m  *manifest.Manifest
	cc conflict.Collector

	silent bool
	debug  bool

	resolvableEntries   map[string]map[string]PropertyEntry // 合并 YAML 和 Properties
	unresolvableEntries map[string]map[string]PropertyEntry // 无法解析的引用

	reservedKeyHandlers map[string]ValueOverride
	reservedProperties  map[string][]ComponentKeyValue

	servletContextPath map[string]string // {componentName: contextPath}

	result ReconcileReport
}

func NewManager(m *manifest.Manifest) *PropertyManager {
	return &PropertyManager{
		m:  m,
		cc: conflict.NewManager(),

		resolvableEntries:   make(map[string]map[string]PropertyEntry),
		unresolvableEntries: make(map[string]map[string]PropertyEntry),

		reservedKeyHandlers: reservedKeyHandlers,
		reservedProperties:  make(map[string][]ComponentKeyValue),

		servletContextPath: make(map[string]string),
	}
}

func (cm *PropertyManager) M() *manifest.Manifest {
	return cm.m
}

func (cm *PropertyManager) Debug() *PropertyManager {
	cm.debug = true
	return cm
}

func (cm *PropertyManager) Silent() *PropertyManager {
	cm.silent = true
	return cm
}

func (cm *PropertyManager) Result() ReconcileReport {
	return cm.result
}

func (pm *PropertyManager) IdentifyPropertiesFileConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(func(ps *PropertyEntry) bool {
		return ps.IsProperties()
	})
}

func (pm *PropertyManager) IdentifyYamlFileConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(func(ps *PropertyEntry) bool {
		return ps.IsYAML()
	})
}

// 合并 YAML 和 Properties 的冲突
func (pm *PropertyManager) identifyAllConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(nil)
}

func (pm *PropertyManager) identifyConflicts(fileTypeFilter func(*PropertyEntry) bool) map[string]map[string]interface{} {
	conflicts := make(map[string]map[string]interface{})
	configPropConflicts := make(map[string]bool)

	// 第一遍：识别所有冲突和 ConfigurationProperties 冲突
	for key := range pm.getAllUniqueKeys() {
		componentValues := make(map[string]interface{})
		var firstValue interface{}
		isConflict := false

		for component, entries := range pm.resolvableEntries {
			if entry, exists := entries[key]; exists && (fileTypeFilter == nil || fileTypeFilter(&entry)) {
				componentValues[component] = entry.Value
				if firstValue == nil {
					firstValue = entry.Value
				} else if !reflect.DeepEqual(firstValue, entry.Value) {
					isConflict = true
				}
			}
		}

		if isConflict && len(componentValues) > 1 {
			conflicts[key] = componentValues

			// 检查是否是 ConfigurationProperties 的一部分
			if integralKey := pm.getConfigurationPropertiesPrefix(key); integralKey != "" {
				log.Printf("@ConfigurationProperties(%s) key[%s] encounters conflict values", integralKey, key)
				configPropConflicts[integralKey] = true
			}
		}
	}

	// 第二遍：将所有 ConfigurationProperties 相关的键标记为冲突
	for key := range pm.getAllUniqueKeys() {
		for integralKey := range configPropConflicts {
			if strings.HasPrefix(key, integralKey) {
				componentValues := make(map[string]interface{})
				for component, entries := range pm.resolvableEntries {
					if entry, exists := entries[key]; exists {
						componentValues[component] = entry.Value
					}
				}
				if len(componentValues) > 0 {
					conflicts[key] = componentValues
				}
				break
			}
		}
	}

	return conflicts
}

func (pm *PropertyManager) getConfigurationPropertiesPrefix(key string) string {
	if pm.m == nil {
		return ""
	}

	for _, prefix := range pm.m.Main.Reconcile.Resources.Property.ConfigurationPropertiesKeys {
		if strings.HasPrefix(key, prefix) {
			return prefix
		}
	}
	return ""
}

func (pm *PropertyManager) getAllUniqueKeys() map[string]struct{} {
	keys := make(map[string]struct{})
	for _, entries := range pm.resolvableEntries {
		for key := range entries {
			keys[key] = struct{}{}
		}
	}
	return keys
}

func (pm *PropertyManager) Resolve(key string) interface{} {
	for _, entries := range pm.resolvableEntries {
		if entry, ok := entries[key]; ok {
			return entry.Value
		}
	}

	return nil
}

// 自动获取 line 里的属性引用占位符，并解析对应属性值，返回解析后的 line
// 如果没有占位符，则返回原 line
func (pm *PropertyManager) ResolveLine(line string) string {
	// 使用正则表达式找到所有的占位符
	matches := P.placeholderRegex.FindAllStringSubmatchIndex(line, -1)

	// 如果没有匹配项，直接返回原始行
	if len(matches) == 0 {
		return line
	}

	// 创建一个新的字符串构建器
	var result strings.Builder
	lastIndex := 0

	for _, match := range matches {
		// 添加占位符之前的文本
		result.WriteString(line[lastIndex:match[0]])

		// 提取占位符中的键
		key := line[match[2]:match[3]]

		// 解析键对应的值
		value := pm.Resolve(key)

		// 如果找到了值，添加到结果中；否则保留原始占位符
		if value != nil {
			result.WriteString(fmt.Sprintf("%v", value))
		} else {
			result.WriteString(line[match[0]:match[1]])
		}

		lastIndex = match[1]
	}

	// 添加最后一个占位符之后的文本
	result.WriteString(line[lastIndex:])

	return result.String()
}
