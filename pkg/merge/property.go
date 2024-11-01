package merge

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/manifest"
	"gopkg.in/yaml.v2"
)

type PropertyManager struct {
	m *manifest.Manifest

	propertyReferences []PropertyReference

	yamlConflictKeys map[string]map[string]interface{} // [key][componentName]
	mergedYaml       map[string]interface{}

	propertiesConflictKeys map[string]map[string]interface{}
	mergedProperties       map[string]interface{}

	allProperties map[string]interface{} // 合并 YAML 和 Properties

	// 属性文件的扩展名有哪些
	propertySourceExts map[string]struct{}

	reservedYamlKeys map[string]ValueOverride
	reservedValues   map[string][]ComponentKeyValue

	servletContextPath  map[string]string // {componentName: contextPath}
	requestMappingRegex *regexp.Regexp
}

func NewPropertyManager(m *manifest.Manifest) *PropertyManager {
	return &PropertyManager{
		yamlConflictKeys:       make(map[string]map[string]interface{}),
		propertiesConflictKeys: make(map[string]map[string]interface{}),
		mergedYaml:             make(map[string]interface{}),
		mergedProperties:       make(map[string]interface{}),
		propertySourceExts: map[string]struct{}{
			".properties": {},
		},
		reservedYamlKeys:    reservedKeyHandlers,
		reservedValues:      make(map[string][]ComponentKeyValue),
		servletContextPath:  make(map[string]string),
		requestMappingRegex: regexp.MustCompile(`(@RequestMapping\s*\(\s*(?:value\s*=)?\s*")([^"]+)("\s*\))`),
		m:                   m,
	}
}

// 扫描 application.yml 以及 application-{profile}.yml，发现冲突的keys
func (cm *PropertyManager) AnalyzeApplicationYamlFiles() error {
	for _, component := range cm.m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			sourceDir := component.SrcDir(baseDir)
			if err := cm.analyzeYamlFile(filepath.Join(sourceDir, "application.yml"), component.SpringProfile, component); err != nil {
				return err
			}
			if component.SpringProfile != "" {
				if err := cm.analyzeYamlFile(filepath.Join(sourceDir, "application-"+component.SpringProfile+".yml"), component.SpringProfile, component); err != nil {
					return err
				}
			}
		}
	}

	cm.applyReservedPropertyRules()
	return nil
}

// 扫描指定的 properties，发现冲突 keys
func (cm *PropertyManager) AnalyzePropertyFiles() error {
	for _, component := range cm.m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			for _, propertySource := range component.Resources.PropertySources {
				ext := filepath.Ext(propertySource)
				if _, present := cm.propertySourceExts[ext]; !present {
					return fmt.Errorf("Invalid property source: %s", propertySource)
				}

				sourceDir := component.SrcDir(baseDir)
				if err := cm.analyzePropertiesFile(filepath.Join(sourceDir, propertySource), component); err != nil {
					return err
				}
			}

		}
	}

	cm.applyReservedPropertyRules()
	return nil
}

// 合并 YAML 和 Properties 的冲突
func (cm *PropertyManager) IdentifyAllPropertyConflicts() map[string]map[string]interface{} {
	allConflicts := make(map[string]map[string]interface{})
	for k, v := range cm.IdentifyYamlFileConflicts() {
		allConflicts[k] = v
	}
	for k, v := range cm.IdentifyPropertiesFileConflicts() {
		allConflicts[k] = v
	}
	return allConflicts
}

func (cm *PropertyManager) IdentifyPropertiesFileConflicts() map[string]map[string]interface{} {
	return cm.identifyPropertyConflicts(cm.propertiesConflictKeys)
}

func (cm *PropertyManager) IdentifyYamlFileConflicts() map[string]map[string]interface{} {
	return cm.identifyPropertyConflicts(cm.yamlConflictKeys)
}

func (cm *PropertyManager) updateRequestMappingInFile(content, contextPath string) string {
	return cm.requestMappingRegex.ReplaceAllStringFunc(content, func(match string) string {
		submatches := cm.requestMappingRegex.FindStringSubmatch(match)
		if len(submatches) == 4 {
			oldPath := submatches[2]
			newPath := filepath.Join(contextPath, oldPath)
			return submatches[1] + newPath + submatches[3]
		}
		return match
	})
}

func (cm *PropertyManager) createJavaRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@Value\s*\(\s*"\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"\s*\)`)
}

func (cm *PropertyManager) createXmlRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(value|key)="\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func (cm *PropertyManager) replaceKeyInMatch(match, key, prefix string) string {
	return strings.Replace(match, "${"+key, "${"+prefix+key, 1)
}

func (cm *PropertyManager) GenerateMergedYamlFile(targetFile string) {
	// merge with propertySettlement
	for key, val := range cm.m.Main.Reconcile.Resources.PropertySettlement {
		log.Printf("propertySettlement %s:%v", key, val)
		cm.mergedYaml[key] = val
	}

	// write file
	data, err := yaml.Marshal(cm.mergedYaml)
	if err != nil {
		log.Fatalf("Error marshalling merged config: %v", err)
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		log.Fatalf("Error creating directory for merged config: %v", err)
	}

	if err := os.WriteFile(targetFile, data, 0644); err != nil {
		log.Fatalf("Error writing merged config to %s: %v", targetFile, err)
	}
}

func (cm *PropertyManager) GenerateMergedPropertiesFile(targetFile string) {
	var builder strings.Builder
	for key, value := range cm.mergedProperties {
		builder.WriteString(fmt.Sprintf("%s=%v\n", key, value))
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		log.Fatalf("Error creating directory for merged properties: %v", err)
	}

	if err := os.WriteFile(targetFile, []byte(builder.String()), 0644); err != nil {
		log.Fatalf("Error writing merged properties to %s: %v", targetFile, err)
	}
}

