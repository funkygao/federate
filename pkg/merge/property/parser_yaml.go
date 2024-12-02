package property

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
	"gopkg.in/yaml.v2"
)

type yamlParser struct{}

func (y *yamlParser) Parse(filePath string, component manifest.ComponentInfo, pm *PropertyManager) error {
	return y.recursiveParseFile(filePath, component.SpringProfile, component, pm)
}

func (y *yamlParser) recursiveParseFile(filePath string, springProfile string, component manifest.ComponentInfo, pm *PropertyManager) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	if !pm.silent || pm.debug {
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
	y.flattenYamlMap(config, "", flatConfig)

	// 注册
	for key, value := range flatConfig {
		pm.r.AddProperty(component, key, value, filePath)
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
						if !pm.silent {
							log.Printf("[%s:%s] Following spring.profiles.include: %s", component.Name, springProfile, trimmedProfile)
						}

						// 递归
						y.recursiveParseFile(filepath.Join(filepath.Dir(filePath), "application-"+trimmedProfile+".yml"), includeProfile, component, pm)
					}
				}
			}
		}
	}

	return nil
}

func (y *yamlParser) flattenYamlMap(data map[interface{}]interface{}, parentKey string, result map[string]interface{}) {
	for k, v := range data {
		fullKey := strings.TrimPrefix(fmt.Sprintf("%s.%v", parentKey, k), ".")
		switch vTyped := v.(type) {
		case map[interface{}]interface{}:
			y.flattenYamlMap(vTyped, fullKey, result)
		default:
			result[fullKey] = v
		}
	}
}

func (y *yamlParser) Generate(entries map[string]PropertyEntry, rawKeys []string, targetFile string) error {
	mergedYaml := make(map[string]interface{})
	for key, entry := range entries {
		if entry.WasReference() {
			mergedYaml[key] = entry.Raw
		} else {
			mergedYaml[key] = entry.Value
		}
	}

	// 序列化
	data, err := yaml.Marshal(mergedYaml)
	if err != nil {
		return err
	}

	// 对 YAML 数据进行后处理，移除 raw 值周围的单引号
	yamlString := string(data)
	for _, rawKey := range rawKeys {
		value := mergedYaml[rawKey]
		if rawValue, ok := value.(string); ok {
			oldQuotedValue := "'" + rawValue + "'"
			yamlString = strings.Replace(yamlString, oldQuotedValue, rawValue, -1)
		} else {
			log.Printf("Raw key[%s] value is not string, ignored", rawKey)
		}
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		return err
	}

	// Write file
	if err := os.WriteFile(targetFile, []byte(yamlString), 0644); err != nil {
		return err
	}

	return nil
}
