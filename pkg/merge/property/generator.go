package property

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
		for key, entry := range cm.resolvableEntries[component.Name] {
			if entry.IsYAML() && !processedKeys[key] {
				if entry.WasReference() {
					mergedYaml[key] = entry.RawString
				} else {
					// 否则使用解析后的值
					mergedYaml[key] = entry.Value
				}
				processedKeys[key] = true
			}
		}
	}

	// 序列化
	data, err := yaml.Marshal(mergedYaml)
	if err != nil {
		log.Fatalf("Error marshalling merged config: %v", err)
	}

	// 对 YAML 数据进行后处理，移除 raw 值周围的单引号
	yamlString := string(data)
	for _, rawKey := range cm.m.Main.Reconcile.Resources.Property.RawKeys {
		value := mergedYaml[rawKey]
		if rawValue, ok := value.(string); ok {
			oldQuotedValue := "'" + rawValue + "'"
			yamlString = strings.Replace(yamlString, oldQuotedValue, rawValue, -1)
		} else {
			log.Printf("Raw key[%s] value is not string, ignored", rawKey)
		}
	}

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		log.Fatalf("Error creating directory for merged config: %v", err)
	}

	// Write file
	if err := os.WriteFile(targetFile, []byte(yamlString), 0644); err != nil {
		log.Fatalf("Error writing merged config to %s: %v", targetFile, err)
	}

	log.Printf("Generated %s", targetFile)
}

func (cm *PropertyManager) GenerateMergedPropertiesFile(targetFile string) {
	var builder strings.Builder

	// 使用一个 set 来跟踪已处理的 key
	processedKeys := make(map[string]bool)

	// Merge all resolved properties
	for _, entries := range cm.resolvableEntries {
		for key, entry := range entries {
			if entry.IsProperties() && !processedKeys[key] {
				// TODO not use RawString?
				builder.WriteString(fmt.Sprintf("%s=%v\n", key, entry.Value))
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
