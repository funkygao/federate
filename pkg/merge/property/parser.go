package property

import (
	"bufio"
	"federate/pkg/manifest"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
)

// PropertyParser 定义了属性文件解析器的接口
type PropertyParser interface {
	Parse(filePath string, component manifest.ComponentInfo, cm *PropertyManager) error
}

type yamlParser struct{}

func (y *yamlParser) Parse(filePath string, component manifest.ComponentInfo, pm *PropertyManager) error {
	return y.parseFile(filePath, component.SpringProfile, component, pm)
}

func (y *yamlParser) parseFile(filePath string, springProfile string, component manifest.ComponentInfo, pm *PropertyManager) error {
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
						y.parseFile(filepath.Join(filepath.Dir(filePath), "application-"+trimmedProfile+".yml"), includeProfile, component, pm)
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

type propertiesParser struct{}

func (p *propertiesParser) Parse(filePath string, component manifest.ComponentInfo, cm *PropertyManager) error {
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

			cm.r.AddProperty(component, key, value, filePath)
		}
	}

	if err = scanner.Err(); err != nil {
		return err
	}

	return nil
}
