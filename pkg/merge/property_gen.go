package merge

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
)

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
