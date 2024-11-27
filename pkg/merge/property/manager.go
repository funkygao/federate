package property

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/util"
)

type ReconcileReport struct {
	KeyPrefixed             int
	RequestMapping          int
	ConfigurationProperties int
}

// 合并 YAML, properties 文件
type PropertyManager struct {
	m *manifest.Manifest
	r *registry

	silent bool
	debug  bool

	servletContextPath map[string]string // {componentName: contextPath}

	result ReconcileReport
}

func NewManager(m *manifest.Manifest) *PropertyManager {
	pm := &PropertyManager{
		m:                  m,
		r:                  newRegistry(m, false),
		servletContextPath: make(map[string]string),
	}
	pm.r.pm = pm
	return pm
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
	cm.r.silent = true
	return cm
}

func (cm *PropertyManager) Result() ReconcileReport {
	return cm.result
}

// 分析 .yml & .properties
func (cm *PropertyManager) Analyze() error {
	for _, component := range cm.m.Components {
		if err := cm.analyzeComponent(component); err != nil {
			return err
		}
	}

	// 解析所有引用
	cm.resolveAllReferences()

	// 应用保留key处理规则
	cm.applyReservedPropertyRules()
	return nil
}

func (cm *PropertyManager) analyzeComponent(component manifest.ComponentInfo) error {
	for _, baseDir := range component.Resources.BaseDirs {
		sourceDir := component.SrcDir(baseDir)

		// 分析 application.yml 和 application-{profile}.yml
		propertyFiles := []string{"application.yml"}
		if component.SpringProfile != "" {
			propertyFiles = append(propertyFiles, "application-"+component.SpringProfile+".yml")
		}
		// 加上用户指定的资源文件
		propertyFiles = append(propertyFiles, component.Resources.PropertySources...)

		for _, propertyFile := range propertyFiles {
			filePath := filepath.Join(sourceDir, propertyFile)
			if !util.FileExists(filePath) {
				log.Printf("[%s] Not found: %s", component.Name, filePath)
				continue
			}

			parser, supported := ParserByFile(propertyFile)
			if !supported {
				return fmt.Errorf("unsupported file type: %s", filePath)
			}

			if err := parser.Parse(filePath, component, cm); err != nil {
				return err
			}
		}
	}

	return nil
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

		for component, entries := range pm.r.resolvableEntries {
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
				for component, entries := range pm.r.resolvableEntries {
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
	for _, entries := range pm.r.resolvableEntries {
		for key := range entries {
			keys[key] = struct{}{}
		}
	}
	return keys
}

func (pm *PropertyManager) ContainsKey(c manifest.ComponentInfo, key string) bool {
	_, ok := pm.r.getComponentProperty(c, key)
	return ok
}

func (pm *PropertyManager) Resolve(key string) interface{} {
	for _, entries := range pm.r.resolvableEntries {
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
