package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

var systemGetPropertyRegex = regexp.MustCompile(`System\.getProperty\s*\(\s*([^)]+)\s*\)`)

func ReconcileEnvConflicts(m *manifest.Manifest) {
	propertyKeys := make(map[string]struct{})

	for _, component := range m.Components {
		err := filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if java.IsJavaMainSource(info, path) {
				keys, err := findSystemGetPropertyKeys(path)
				if err != nil {
					log.Printf("Error processing file %s: %v", path, err)
					return nil
				}
				for _, key := range keys {
					propertyKeys[key] = struct{}{}
				}
			}

			return nil
		})

		if err != nil {
			log.Printf("Error walking the path %s: %v", component.RootDir(), err)
		}
	}

	if len(propertyKeys) > 0 {
		log.Println("System.getProperty keys found:")
		for key := range propertyKeys {
			log.Printf("  - %s", key)
		}
	} else {
		log.Println("System.getProperty is OK")
	}
}

func findSystemGetPropertyKeys(filePath string) ([]string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	matches := systemGetPropertyRegex.FindAllSubmatch(content, -1)

	keys := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			arg := strings.TrimSpace(string(match[1]))
			if strings.HasPrefix(arg, "\"") && strings.HasSuffix(arg, "\"") {
				// 直接的字符串字面量
				keys = append(keys, arg[1:len(arg)-1])
			} else if strings.HasPrefix(arg, "'") && strings.HasSuffix(arg, "'") {
				// 使用单引号的字符串字面量
				keys = append(keys, arg[1:len(arg)-1])
			} else {
				// 可能是变量或常量
				keys = append(keys, "VARIABLE: "+arg)
			}
		}
	}

	return keys, nil
}
