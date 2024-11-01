package merge

import (
	"bufio"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
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

func (cm *PropertyManager) handleFileErr(componentName, filePath string, err error) error {
	if _, ok := err.(*fs.PathError); ok {
		return nil
	}
	return err
}

func (cm *PropertyManager) analyzePropertiesFile(filePath string, component manifest.ComponentInfo) error {
	file, err := os.Open(filePath)
	if err != nil {
		return cm.handleFileErr(component.Name, filePath, err)
	}
	defer file.Close()

	log.Printf("[%s] Processing %s", component.Name, filePath)

	properties := make(map[string]interface{})
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			properties[key] = value
		}
	}

	if err = scanner.Err(); err != nil {
		return err
	}

	mergeMaps(cm.mergedProperties, properties, cm.propertiesConflictKeys, cm, component)
	return nil
}

func (cm *PropertyManager) analyzeYamlFile(filePath string, springProfile string, component manifest.ComponentInfo) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return cm.handleFileErr(component.Name, filePath, err)
	}

	log.Printf("[%s:%s] Processing %s", component.Name, springProfile, filePath)

	dataStr := string(data)
	dataStr = strings.ReplaceAll(dataStr, springProfileActive, springProfile)

	var config map[interface{}]interface{}
	if err = yaml.Unmarshal([]byte(dataStr), &config); err != nil {
		return err
	}

	flatConfig := make(map[string]interface{})
	cm.flattenYamlMap(config, "", flatConfig)

	mergeMaps(cm.mergedYaml, flatConfig, cm.yamlConflictKeys, cm, component)

	// 处理 spring.profiles.include
	if includes, ok := flatConfig[springProfileInclude]; ok {
		if includeProfiles, ok := includes.(string); ok {
			for _, includeProfile := range strings.Split(includeProfiles, ",") {
				includeProfile = strings.TrimSpace(includeProfile)
				log.Printf("[%s:%s] Detected spring.profiles.include: %s", component.Name, springProfile, includeProfile)
				if includeProfile != "" {
					cm.analyzeYamlFile(filepath.Join(filepath.Dir(filePath), "application-"+includeProfile+".yml"), includeProfile, component)
				}
			}
		}
	}

	return nil
}

func (cm *PropertyManager) recordConflict(conflictMap map[string]map[string]interface{}, key, componentName string, value interface{}) {
	if _, exists := conflictMap[key]; !exists {
		conflictMap[key] = make(map[string]interface{})
	}
	conflictMap[key][componentName] = trimValue(value)
}

func (cm *PropertyManager) identifyPropertyConflicts(conflictKeys map[string]map[string]interface{}) map[string]map[string]interface{} {
	filteredConflicts := make(map[string]map[string]interface{})
	for key, components := range conflictKeys {
		if len(components) == 1 {
			// 如果只在一个组件中出现，不算冲突
			continue
		}

		uniqueValues := make(map[string]bool)
		for _, value := range components {
			if !isEmptyValue(value) {
				uniqueValues[fmt.Sprintf("%v", value)] = true
			}
		}

		if len(uniqueValues) > 1 {
			filteredConflicts[key] = components
		}
	}
	return filteredConflicts
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

func (cm *PropertyManager) flattenYamlMap(data map[interface{}]interface{}, parentKey string, result map[string]interface{}) {
	for k, v := range data {
		fullKey := strings.TrimPrefix(fmt.Sprintf("%s.%v", parentKey, k), ".")
		switch vTyped := v.(type) {
		case map[interface{}]interface{}:
			cm.flattenYamlMap(vTyped, fullKey, result)
		default:
			result[fullKey] = v
		}
	}
}

func (cm *PropertyManager) unflattenYamlMap(data map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for key, value := range data {
		keys := strings.Split(key, ".")
		currentMap := result
		for i, k := range keys {
			if i == len(keys)-1 {
				currentMap[k] = value
			} else {
				if _, ok := currentMap[k]; !ok {
					currentMap[k] = make(map[string]interface{})
				}
				currentMap = currentMap[k].(map[string]interface{})
			}
		}
	}
	return result
}

func mergeMaps(dest, src map[string]interface{}, conflictMap map[string]map[string]interface{}, cm *PropertyManager, component manifest.ComponentInfo) {
	for k, v := range src {
		if !cm.isReservedProperty(k) {
			cm.registerPropertyValue(conflictMap, k, component.Name, v)
		} else {
			cm.registerReservedProperty(k, component, v)
		}

		dest[k] = v
	}
}

// 记录值并检查冲突
func (cm *PropertyManager) registerPropertyValue(conflictMap map[string]map[string]interface{}, key, componentName string, value interface{}) {
	if _, exists := conflictMap[key]; !exists {
		conflictMap[key] = make(map[string]interface{})
	}
	conflictMap[key][componentName] = value
}

func (cm *PropertyManager) registerReservedProperty(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := cm.reservedValues[key]; !exists {
		cm.reservedValues[key] = []ComponentKeyValue{}
	}
	cm.reservedValues[key] = append(cm.reservedValues[key], ComponentKeyValue{Component: component, Value: value})
}

func (cm *PropertyManager) applyReservedPropertyRules() {
	var cellData [][]string
	for key, values := range cm.reservedValues {
		if handler, exists := cm.reservedYamlKeys[key]; exists {
			if cm.m.Main.Reconcile.PropertySettled(key) {
				color.Yellow("key:%s reserved, but used directive: propertySettled, skipped", key)
				continue
			}
			if value := handler(cm, values); value != nil {
				cm.mergedYaml[key] = value
				cellData = append(cellData, []string{key, fmt.Sprintf("%v", value)})
			} else {
				delete(cm.mergedYaml, key)
				cellData = append(cellData, []string{key, color.New(color.FgRed).Add(color.CrossedOut).Sprintf("deleted")})
			}
		}
	}

	header := []string{"Reserved Key", "Value"}
	tablerender.DisplayTable(header, cellData, false, -1)
}

func isEmptyValue(value interface{}) bool {
	switch v := value.(type) {
	case nil:
		return true
	case string:
		return v == ""
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return v == 0
	case bool:
		return !v
	case []interface{}:
		return len(v) == 0
	case map[string]interface{}:
		return len(v) == 0
	default:
		return false
	}
}

func trimValue(value interface{}) interface{} {
	if str, ok := value.(string); ok {
		return strings.TrimSpace(str)
	}
	return value
}

func (cm *PropertyManager) isReservedProperty(key string) bool {
	_, exists := cm.reservedYamlKeys[key]
	return exists
}
