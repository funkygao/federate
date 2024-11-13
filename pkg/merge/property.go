package merge

import (
	"fmt"
	"path/filepath"
	"reflect"
	"regexp"

	"federate/pkg/manifest"
	"federate/pkg/util"
)

type PropertyManager struct {
	m *manifest.Manifest

	resolvedProperties   map[string]map[string]PropertySource // 合并 YAML 和 Properties
	unresolvedProperties map[string]map[string]PropertySource // 无法解析的引用

	// 属性文件的扩展名有哪些被支持
	propertySourceExts map[string]struct{}

	reservedYamlKeys map[string]ValueOverride
	reservedValues   map[string][]ComponentKeyValue

	servletContextPath  map[string]string // {componentName: contextPath}
	requestMappingRegex *regexp.Regexp
}

func NewPropertyManager(m *manifest.Manifest) *PropertyManager {
	return &PropertyManager{
		m: m,

		resolvedProperties:   make(map[string]map[string]PropertySource),
		unresolvedProperties: make(map[string]map[string]PropertySource),

		propertySourceExts: map[string]struct{}{
			".properties": {},
			".yml":        {},
			".yaml":       {},
		},

		reservedYamlKeys: reservedKeyHandlers,
		reservedValues:   make(map[string][]ComponentKeyValue),

		servletContextPath:  make(map[string]string),
		requestMappingRegex: regexp.MustCompile(`(@RequestMapping\s*\(\s*(?:value\s*=)?\s*")([^"]+)("\s*\))`),
	}
}

func (cm *PropertyManager) AnalyzeAllPropertySources() error {
	for _, component := range cm.m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			sourceDir := component.SrcDir(baseDir)

			// 分析 application.yml 和 application-{profile}.yml
			yamlFiles := []string{"application.yml"}
			if component.SpringProfile != "" {
				yamlFiles = append(yamlFiles, "application-"+component.SpringProfile+".yml")
			}
			for _, f := range yamlFiles {
				filePath := filepath.Join(sourceDir, f)
				if !util.FileExists(filePath) {
					continue
				}
				if err := cm.analyzeYamlFile(filePath, component.SpringProfile, component); err != nil {
					return err
				}
			}

			// 分析其他属性文件，基本上就是 .properties
			for _, propertySource := range component.Resources.PropertySources {
				filePath := filepath.Join(sourceDir, propertySource)
				if !util.FileExists(filePath) {
					continue
				}
				ext := filepath.Ext(propertySource)
				if _, present := cm.propertySourceExts[ext]; !present {
					return fmt.Errorf("Invalid property source: %s", propertySource)
				}

				switch ext {
				case ".yml", ".yaml":
					if err := cm.analyzeYamlFile(filePath, component.SpringProfile, component); err != nil {
						return err
					}
				case ".properties":
					if err := cm.analyzePropertiesFile(filePath, component); err != nil {
						return err
					}
				default:
					return fmt.Errorf("Unsupported file type: %s", filePath)
				}
			}
		}
	}

	// 解析所有引用
	cm.resolveAllReferences()

	// 应用保留key处理规则
	cm.applyReservedPropertyRules()
	return nil
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
		}
	}
	return conflicts
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
