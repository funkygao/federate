package merge

import (
	"bufio"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"federate/pkg/concurrent"
	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/fatih/color"
	"gopkg.in/yaml.v2"
)

type PropertySourcesManager struct {
	m *manifest.Manifest

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

func NewPropertySourcesManager(m *manifest.Manifest) *PropertySourcesManager {
	return &PropertySourcesManager{
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
func (cm *PropertySourcesManager) PrepareMergeApplicationYaml() error {
	for _, component := range cm.m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			sourceDir := component.SrcDir(baseDir)
			if err := cm.planMergeYamlFile(filepath.Join(sourceDir, "application.yml"), component.SpringProfile, component); err != nil {
				return err
			}
			if component.SpringProfile != "" {
				if err := cm.planMergeYamlFile(filepath.Join(sourceDir, "application-"+component.SpringProfile+".yml"), component.SpringProfile, component); err != nil {
					return err
				}
			}
		}
	}

	cm.finalizeReservedKeys()
	return nil
}

// 扫描指定的 properties，发现冲突 keys
func (cm *PropertySourcesManager) PrepareMergePropertiesFiles() error {
	for _, component := range cm.m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			for _, propertySource := range component.Resources.PropertySources {
				ext := filepath.Ext(propertySource)
				if _, present := cm.propertySourceExts[ext]; !present {
					return fmt.Errorf("Invalid property source: %s", propertySource)
				}

				sourceDir := component.SrcDir(baseDir)
				if err := cm.planMergePropertiesFile(filepath.Join(sourceDir, propertySource), component); err != nil {
					return err
				}
			}

		}
	}

	cm.finalizeReservedKeys()
	return nil
}

// 合并 YAML 和 Properties 的冲突
func (cm *PropertySourcesManager) GetAllConflicts() map[string]map[string]interface{} {
	allConflicts := make(map[string]map[string]interface{})
	for k, v := range cm.GetYamlConflicts() {
		allConflicts[k] = v
	}
	for k, v := range cm.GetPropertiesConflicts() {
		allConflicts[k] = v
	}
	return allConflicts
}

func (cm *PropertySourcesManager) GetPropertiesConflicts() map[string]map[string]interface{} {
	return cm.getConflicts(cm.propertiesConflictKeys)
}

func (cm *PropertySourcesManager) GetYamlConflicts() map[string]map[string]interface{} {
	return cm.getConflicts(cm.yamlConflictKeys)
}

type PropertySourcesReconciled struct {
	KeyPrefixed    int
	RequestMapping int
}

// 调和冲突：
func (cm *PropertySourcesManager) ReconcileConflicts(dryRun bool) (result PropertySourcesReconciled, err error) {
	conflictKeys := cm.GetYamlConflicts()
	if len(conflictKeys) == 0 {
		return
	}

	// Group keys by component
	componentKeys := make(map[string][]string)
	var cellData [][]string
	for key, components := range conflictKeys {
		for componentName, value := range components {
			componentKeys[componentName] = append(componentKeys[componentName], key)

			prefixedKey := cm.componentKeyPrefix(componentName) + key
			if value == nil {
				value = ""
			}
			cm.mergedYaml[prefixedKey] = value
			//delete(cm.mergedYaml, key) 原有的key不能删除：第三方包内部，可能在使用该 key

			cellData = append(cellData, []string{prefixedKey, util.Truncate(fmt.Sprintf("%v", value), 60)})
		}
	}

	header := []string{"New Key", "Value"}
	tablerender.DisplayTable(header, cellData, false, -1)
	log.Printf("Reconciled %d conflicting keys into %d keys", len(conflictKeys), len(cellData))

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	for componentName, keys := range componentKeys {
		component := cm.m.ComponentByName(componentName)
		prefix := cm.componentKeyPrefix(componentName)

		executor.AddTask(&reconcileTask{
			cm:                 cm,
			component:          component,
			keys:               keys,
			prefix:             prefix,
			dryRun:             dryRun,
			servletContextPath: cm.servletContextPath[componentName],
			result:             reconcileTaskResult{},
		})
	}

	errors := executor.Execute()
	if len(errors) > 0 {
		err = errors[0] // 返回第一个遇到的错误
	}

	for _, task := range executor.Tasks() {
		reconcileTask := task.(*reconcileTask)
		result.KeyPrefixed += reconcileTask.result.keyPrefixed
		result.RequestMapping += reconcileTask.result.requestMapping
	}

	return
}

func (cm *PropertySourcesManager) componentKeyPrefix(componentName string) string {
	return componentName + "."
}

func (cm *PropertySourcesManager) updateRequestMappingInFile(content, contextPath string) string {
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

func (cm *PropertySourcesManager) createJavaRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@Value\s*\(\s*"\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"\s*\)`)
}

func (cm *PropertySourcesManager) createXmlRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(value|key)="\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func (cm *PropertySourcesManager) replaceKeyInMatch(match, key, prefix string) string {
	return strings.Replace(match, "${"+key, "${"+prefix+key, 1)
}

func (cm *PropertySourcesManager) handleFileErr(componentName, filePath string, err error) error {
	if _, ok := err.(*fs.PathError); ok {
		return nil
	}
	return err
}

func (cm *PropertySourcesManager) planMergePropertiesFile(filePath string, component manifest.ComponentInfo) error {
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

func (cm *PropertySourcesManager) planMergeYamlFile(filePath string, springProfile string, component manifest.ComponentInfo) error {
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
					cm.planMergeYamlFile(filepath.Join(filepath.Dir(filePath), "application-"+includeProfile+".yml"), includeProfile, component)
				}
			}
		}
	}

	return nil
}

func (cm *PropertySourcesManager) recordConflict(conflictMap map[string]map[string]interface{}, key, componentName string, value interface{}) {
	if _, exists := conflictMap[key]; !exists {
		conflictMap[key] = make(map[string]interface{})
	}
	conflictMap[key][componentName] = trimValue(value)
}

func (cm *PropertySourcesManager) getConflicts(conflictKeys map[string]map[string]interface{}) map[string]map[string]interface{} {
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

func (cm *PropertySourcesManager) WriteMergedYaml(targetFile string) {
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

func (cm *PropertySourcesManager) WriteMergedProperties(targetFile string) {
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

func (cm *PropertySourcesManager) flattenYamlMap(data map[interface{}]interface{}, parentKey string, result map[string]interface{}) {
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

func (cm *PropertySourcesManager) unflattenYamlMap(data map[string]interface{}) map[string]interface{} {
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

func mergeMaps(dest, src map[string]interface{}, conflictMap map[string]map[string]interface{}, cm *PropertySourcesManager, component manifest.ComponentInfo) {
	for k, v := range src {
		if !cm.isReservedKey(k) {
			cm.recordValue(conflictMap, k, component.Name, v)
		} else {
			cm.recordReservedValue(k, component, v)
		}

		dest[k] = v
	}
}

// 记录值并检查冲突
func (cm *PropertySourcesManager) recordValue(conflictMap map[string]map[string]interface{}, key, componentName string, value interface{}) {
	if _, exists := conflictMap[key]; !exists {
		conflictMap[key] = make(map[string]interface{})
	}
	conflictMap[key][componentName] = value
}

func (cm *PropertySourcesManager) recordReservedValue(key string, component manifest.ComponentInfo, value interface{}) {
	if _, exists := cm.reservedValues[key]; !exists {
		cm.reservedValues[key] = []ComponentKeyValue{}
	}
	cm.reservedValues[key] = append(cm.reservedValues[key], ComponentKeyValue{Component: component, Value: value})
}

func (cm *PropertySourcesManager) finalizeReservedKeys() {
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

func (cm *PropertySourcesManager) isReservedKey(key string) bool {
	_, exists := cm.reservedYamlKeys[key]
	return exists
}
