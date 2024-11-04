package merge

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"github.com/fatih/color"
	"gopkg.in/yaml.v2"
)

func (cm *PropertyManager) analyzePropertiesFile(filePath string, component manifest.ComponentInfo) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	log.Printf("[%s] Processing %s", component.Name, filePath)

	if cm.resolvedProperties[component.Name] == nil {
		cm.resolvedProperties[component.Name] = make(map[string]PropertySource)
	}

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

			cm.registerProperty(component, key, value, filePath)
		}
	}

	if err = scanner.Err(); err != nil {
		return err
	}

	return nil
}

func (cm *PropertyManager) analyzeYamlFile(filePath string, springProfile string, component manifest.ComponentInfo) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	log.Printf("[%s:%s] Processing %s", component.Name, springProfile, filePath)

	dataStr := string(data)
	// handling @spring.profiles.active@
	dataStr = strings.ReplaceAll(dataStr, springProfileActive, springProfile)

	var config map[interface{}]interface{}
	if err = yaml.Unmarshal([]byte(dataStr), &config); err != nil {
		return err
	}

	if cm.resolvedProperties[component.Name] == nil {
		cm.resolvedProperties[component.Name] = make(map[string]PropertySource)
	}

	flatConfig := make(map[string]interface{})
	cm.flattenYamlMap(config, "", flatConfig)

	// 捕获属性引用
	for key, value := range flatConfig {
		cm.registerProperty(component, key, value, filePath)
	}

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

func (cm *PropertyManager) registerProperty(component manifest.ComponentInfo, key string, value interface{}, filePath string) {
	if cm.isReservedProperty(key) {
		// 保留字
		cm.registerReservedProperty(key, component, value)
	} else if val, overridden := cm.m.PropertyOverridden(key); overridden {
		// 用户手工指定值
		cm.resolvedProperties[component.Name][key] = PropertySource{
			Value:    val,
			FilePath: filePath,
		}
	} else {
		cm.resolvedProperties[component.Name][key] = PropertySource{
			Value:    value,
			FilePath: filePath,
		}

		// 捕获属性引用
		if str, ok := value.(string); ok && strings.Contains(str, "${") {
			cm.propertyReferences = append(cm.propertyReferences, PropertyReference{
				Component: component.Name,
				Key:       key,
				Value:     str,
				FilePath:  filePath,
			})
		}
	}
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
				for _, componentProps := range cm.resolvedProperties {
					componentProps[key] = PropertySource{
						Value:    value,
						FilePath: "reserved.yml",
					}
				}

				cellData = append(cellData, []string{key, fmt.Sprintf("%v", value)})
			} else {
				for _, componentProps := range cm.resolvedProperties {
					delete(componentProps, key)
				}
				cellData = append(cellData, []string{key, color.New(color.FgRed).Add(color.CrossedOut).Sprintf("deleted")})
			}
		}
	}

	log.Printf("Reserved keys processed:")
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
