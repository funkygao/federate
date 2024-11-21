package property

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"gopkg.in/yaml.v2"
)

func (cm *PropertyManager) analyzePropertiesFile(filePath string, component manifest.ComponentInfo) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	if !cm.silent || cm.debug {
		log.Printf("[%s] Processing %s", component.Name, filePath)
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

	if !cm.silent || cm.debug {
		log.Printf("[%s:%s] Processing %s", component.Name, springProfile, filePath)
	}

	dataStr := string(data)
	// handling @spring.profiles.active@
	dataStr = strings.ReplaceAll(dataStr, springProfileActive, springProfile)

	var config map[interface{}]interface{}
	if err = yaml.Unmarshal([]byte(dataStr), &config); err != nil {
		return err
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
				if includeProfile != "" {
					// includeProfile is comma separated
					for _, profile := range strings.Split(includeProfile, ",") {
						trimmedProfile := strings.TrimSpace(profile)
						if !cm.silent {
							log.Printf("[%s:%s] Following spring.profiles.include: %s", component.Name, springProfile, trimmedProfile)
						}
						cm.analyzeYamlFile(filepath.Join(filepath.Dir(filePath), "application-"+trimmedProfile+".yml"), includeProfile, component)
					}
				}
			}
		}
	}

	return nil
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
