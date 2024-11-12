package merge

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/util"
	"gopkg.in/yaml.v2"
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

func (cm *PropertyManager) GenerateMergedYamlFile(targetFile string) {
	mergedYaml := make(map[string]interface{})

	// 使用一个 set 来跟踪已处理的 key：否则非冲突key被写入多次
	processedKeys := make(map[string]bool)

	// Merge all resolved properties，按照 component 顺序
	for _, component := range cm.m.Components {
		componentProps := cm.resolvedProperties[component.Name]
		for key, propSource := range componentProps {
			if propSource.IsYAML() && !processedKeys[key] {
				if strings.Contains(propSource.OriginalString, "${") {
					// 如果包含引用，使用 OriginalString
					mergedYaml[key] = propSource.OriginalString
				} else {
					// 否则使用解析后的值
					mergedYaml[key] = propSource.Value
				}
				processedKeys[key] = true
			}
		}
	}

	// write file
	data, err := yaml.Marshal(mergedYaml)
	if err != nil {
		log.Fatalf("Error marshalling merged config: %v", err)
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		log.Fatalf("Error creating directory for merged config: %v", err)
	}

	if err := os.WriteFile(targetFile, data, 0644); err != nil {
		log.Fatalf("Error writing merged config to %s: %v", targetFile, err)
	}

	log.Printf("Generated %s", targetFile)
}

func (cm *PropertyManager) GenerateMergedPropertiesFile(targetFile string) {
	var builder strings.Builder

	// 使用一个 set 来跟踪已处理的 key
	processedKeys := make(map[string]bool)

	// Merge all resolved properties
	for _, componentProps := range cm.resolvedProperties {
		for key, propSource := range componentProps {
			if propSource.IsProperties() && !processedKeys[key] {
				builder.WriteString(fmt.Sprintf("%s=%v\n", key, propSource.Value))
				processedKeys[key] = true
			}
		}
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		log.Fatalf("Error creating directory for merged properties: %v", err)
	}

	if err := os.WriteFile(targetFile, []byte(builder.String()), 0644); err != nil {
		log.Fatalf("Error writing merged properties to %s: %v", targetFile, err)
	}

	log.Printf("Generated %s", targetFile)
}
