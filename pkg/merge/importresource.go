package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
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

		if strings.HasPrefix(line, "@ImportResource") {
			newLine, changed := m.processImportResource(line, componentName)
			if changed {
				lines[i] = newLine
				dirty = true
				m.ImportResourceCount++
				log.Printf("[%s] %s %s => %s", componentName, jf.FileBaseName(), line, newLine)
			}
		}
	}

	if dirty {
		return strings.Join(lines, "\n"), true
	}

	return "", false
}

func (m *ImportResourceManager) processImportResource(line, componentName string) (string, bool) {
	return P.importResourcePattern.ReplaceAllStringFunc(line, func(match string) string {
		// 提取所有资源路径
		resourcePaths := regexp.MustCompile(`"([^"]+)"|'([^']+)'`).FindAllStringSubmatch(match, -1)

		newPaths := make([]string, 0, len(resourcePaths))
		for _, path := range resourcePaths {
			resourcePath := path[1]
			if resourcePath == "" {
				resourcePath = path[2] // 如果使用单引号
			}

			// 检查是否以 "classpath:" 开头
			prefix := ""
			if strings.HasPrefix(resourcePath, "classpath:") {
				prefix = "classpath:"
				resourcePath = strings.TrimPrefix(resourcePath, "classpath:")
			}

			// 构造新的资源路径
			newResourcePath := fmt.Sprintf(`"%s%s/%s/%s"`, prefix, federated.FederatedDir, componentName, resourcePath)
			newPaths = append(newPaths, newResourcePath)
		}

		// 重构 @ImportResource 注解
		if len(newPaths) == 1 {
			return fmt.Sprintf(`@ImportResource(%s)`, newPaths[0])
		} else {
			return fmt.Sprintf(`@ImportResource(locations = {%s})`, strings.Join(newPaths, ", "))
		}
	}), true
}
