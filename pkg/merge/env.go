package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

type envManager struct {
	m *manifest.Manifest
}

func NewEnvManager(m *manifest.Manifest) Reconciler {
	return newEnvManager(m)
}

func newEnvManager(m *manifest.Manifest) *envManager {
	return &envManager{m: m}
}

// TODO xml 也可能引用环境变量
func (e *envManager) Reconcile(dryRun bool) error {
	propertyKeys := make(map[string]struct{})

	for _, component := range e.m.Components {
		err := filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if java.IsJavaMainSource(info, path) {
				keys, err := e.findSystemGetPropertyKeys(path)
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
	return nil
}

func (e *envManager) findSystemGetPropertyKeys(filePath string) ([]string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	matches := P.systemGetPropertyRegex.FindAllSubmatch(content, -1)

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
