package property

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
	"gopkg.in/yaml.v2"
)

type yamlParser struct {
	jsonRegexp *regexp.Regexp
}

func newYamlParser() *yamlParser {
	// 使用正则表达式匹配 '{' 开头 '}' 结尾的字符串，包括跨行的情况
	return &yamlParser{
		jsonRegexp: regexp.MustCompile(`(?m)^(\S+):\s*'(\{(?:.|\n)*?\})'`),
	}
}

func (y *yamlParser) Parse(filePath string, component manifest.ComponentInfo, pm *PropertyManager) error {
	return y.recursiveParseFile(filePath, component.SpringProfile, component, pm)
}

func (y *yamlParser) recursiveParseFile(filePath string, springProfile string, component manifest.ComponentInfo, pm *PropertyManager) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	if !pm.silent || pm.debug {
		log.Printf("[%s:%s] Parsing %s", component.Name, springProfile, filePath)
	}

	dataStr := string(data)
	// handling @spring.profiles.active@
	dataStr = strings.ReplaceAll(dataStr, springProfileActive, springProfile)

	var config map[any]any
	if err = yaml.Unmarshal([]byte(dataStr), &config); err != nil {
		return err
	}

	flatConfig := make(map[string]any)
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

func (y *yamlParser) flattenYamlMap(data map[any]any, parentKey string, result map[string]any) {
	for k, v := range data {
		fullKey := strings.TrimPrefix(fmt.Sprintf("%s.%v", parentKey, k), ".")
		switch vTyped := v.(type) {
		case map[any]any:
			y.flattenYamlMap(vTyped, fullKey, result)
		default:
			result[fullKey] = v
		}
	}
}

func (y *yamlParser) Generate(entries map[string]PropertyEntry, targetFile string) error {
	mergedYaml := make(map[string]any)
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
	yamlString = y.applyJSONPatch(yamlString)

	if err := os.MkdirAll(filepath.Dir(targetFile), 0755); err != nil {
		return err
	}

	// Write file
	if err := os.WriteFile(targetFile, []byte(yamlString), 0644); err != nil {
		return err
	}

	return nil
}

func (y *yamlParser) applyJSONPatch(yamlString string) string {
	return y.jsonRegexp.ReplaceAllStringFunc(yamlString, func(match string) string {
		// 提取 key 和 JSON 字符串
		parts := y.jsonRegexp.FindStringSubmatch(match)
		if len(parts) != 3 {
			return match // 如果没有匹配到预期的格式，返回原字符串
		}

		key := parts[1]
		jsonStr := parts[2]

		// 将多行 JSON 合并为一行
		jsonStr = regexp.MustCompile(`\s+`).ReplaceAllString(jsonStr, " ")

		// 记录被替换的 key
		ledger.Get().TransformJsonKey(key)
		log.Printf("[%s/application.yml] Transforming JSON value for key: %s", federated.FederatedDir, key)

		// 返回格式化后的字符串，不包含单引号
		return fmt.Sprintf("%s: %s", key, jsonStr)
	})
}
