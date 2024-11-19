package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/manifest"
)

type ImportResourceManager struct {
	ImportResourceCount int
}

func NewImportResourceManager() *ImportResourceManager {
	return &ImportResourceManager{}
}

func (m *ImportResourceManager) Reconcile(manifest *manifest.Manifest) error {
	for _, component := range manifest.Components {
		if err := m.reconcileComponent(component); err != nil {
			return err
		}
	}
	return nil
}

func (m *ImportResourceManager) reconcileComponent(component manifest.ComponentInfo) error {
	err := filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !java.IsJavaMainSource(info, path) {
			return nil
		}

		fileContent, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}

		javaFile := NewJavaFile(path, &component, string(fileContent))
		newFileContent, dirty := m.reconcileJavaFile(javaFile)

		if dirty {
			err = ioutil.WriteFile(path, []byte(newFileContent), info.Mode())
			if err != nil {
				return err
			}
		}
		return nil
	})
	return err
}

func (m *ImportResourceManager) reconcileJavaFile(jf *JavaFile) (string, bool) {
	componentName := jf.ComponentName()
	commentTracker := NewCommentTracker()
	lines := jf.lines()
	dirty := false

	for i, line := range lines {
		if commentTracker.InComment(line) {
			continue
		}

		matches := P.importResourcePattern.FindStringSubmatch(line)
		if len(matches) > 0 {
			resourcePath := matches[1]
			if resourcePath == "" {
				resourcePath = matches[2] // 如果使用单引号
			}

			// 检查是否以 "classpath:" 开头
			prefix := ""
			if strings.HasPrefix(resourcePath, "classpath:") {
				prefix = "classpath:"
				resourcePath = strings.TrimPrefix(resourcePath, "classpath:")
			}

			// 构造新的资源路径
			newResourcePath := fmt.Sprintf("%s%s/%s/%s", prefix, federated.FederatedDir, componentName, resourcePath)

			// 替换原有的 ImportResource 注解
			newLine := P.importResourcePattern.ReplaceAllString(line, fmt.Sprintf(`@ImportResource("%s")`, newResourcePath))
			log.Printf("[%s] %s %s => %s", componentName, jf.FileBaseName(), line, newLine)

			if newLine != line {
				lines[i] = newLine
				dirty = true
				m.ImportResourceCount++
			}
		}
	}

	if dirty {
		return strings.Join(lines, "\n"), true
	}

	return "", false
}
