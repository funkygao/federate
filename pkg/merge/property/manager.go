package property

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/merge/conflict"
)

type PropertyManager struct {
	m  *manifest.Manifest
	cc conflict.Collector

	silent bool
	debug  bool

	resolvedProperties   map[string]map[string]PropertySource // 合并 YAML 和 Properties
	unresolvedProperties map[string]map[string]PropertySource // 无法解析的引用

	reservedYamlKeys map[string]ValueOverride
	reservedValues   map[string][]ComponentKeyValue

	servletContextPath map[string]string // {componentName: contextPath}
}

func NewManager(m *manifest.Manifest) *PropertyManager {
	return &PropertyManager{
		m:  m,
		cc: conflict.NewManager(),

		resolvedProperties:   make(map[string]map[string]PropertySource),
		unresolvedProperties: make(map[string]map[string]PropertySource),

		reservedYamlKeys: reservedKeyHandlers,
		reservedValues:   make(map[string][]ComponentKeyValue),

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

// 合并 YAML 和 Properties 的冲突
func (pm *PropertyManager) IdentifyAllConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(nil)
}

func (pm *PropertyManager) IdentifyPropertiesFileConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(func(ps *PropertySource) bool {
		return ps.IsProperties()
	})
}

func (pm *PropertyManager) IdentifyYamlFileConflicts() map[string]map[string]interface{} {
	return pm.identifyConflicts(func(ps *PropertySource) bool {
		return ps.IsYAML()
	})
}

func (pm *PropertyManager) identifyConflicts(fileTypeFilter func(*PropertySource) bool) map[string]map[string]interface{} {
	conflicts := make(map[string]map[string]interface{})
	configPropConflicts := make(map[string]bool)

	// 第一遍：识别所有冲突和 ConfigurationProperties 冲突
	for key := range pm.getAllUniqueKeys() {
		componentValues := make(map[string]interface{})
		var firstValue interface{}
		isConflict := false

		for component, props := range pm.resolvedProperties {
			if propSource, exists := props[key]; exists && (fileTypeFilter == nil || fileTypeFilter(&propSource)) {
				componentValues[component] = propSource.Value
				if firstValue == nil {
					firstValue = propSource.Value
				} else if !reflect.DeepEqual(firstValue, propSource.Value) {
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
				for component, props := range pm.resolvedProperties {
					if propSource, exists := props[key]; exists {
						componentValues[component] = propSource.Value
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
	for _, props := range pm.resolvedProperties {
		for key := range props {
			keys[key] = struct{}{}
		}
	}
	return keys
}

func (pm *PropertyManager) Resolve(key string) interface{} {
	for _, props := range pm.resolvedProperties {
		if val, ok := props[key]; ok {
			return val.Value
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
