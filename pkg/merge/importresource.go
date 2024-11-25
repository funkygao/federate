package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/code"
	"federate/pkg/diff"
	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/merge/transformer"
)

// @ImportResource
type ImportResourceManager struct {
	m *manifest.Manifest

	ImportResourceCount int
}

func NewImportResourceManager(m *manifest.Manifest) Reconciler {
	return newImportResourceManager(m)
}

func newImportResourceManager(m *manifest.Manifest) *ImportResourceManager {
	return &ImportResourceManager{m: m}
}

func (m *ImportResourceManager) Reconcile(dryRun bool) error {
	for _, component := range m.m.Components {
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

		if m.m != nil && m.m.Main.MainClass.ExcludeJavaFile(info.Name()) {
			log.Printf("[%s] Excluded %s", component.Name, info.Name())
			return nil
		}

		oldFileConent := string(fileContent)
		javaFile := code.NewJavaFile(path, &component, oldFileConent)
		newFileContent, dirty := m.reconcileJavaFile(javaFile)

		if dirty {
			log.Printf("[%s] %s Updated:", component.Name, info.Name())
			diff.RenderUnifiedDiff(oldFileConent, newFileContent)

			err = ioutil.WriteFile(path, []byte(newFileContent), info.Mode())
			if err != nil {
				return err
			}
		}
		return nil
	})
	return err
}

func (m *ImportResourceManager) reconcileJavaFile(jf *code.JavaFile) (string, bool) {
	componentName := jf.ComponentName()
	commentTracker := code.NewCommentTracker()
	lines := jf.RawLines()
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
				//log.Printf("[%s] %s %s => %s", componentName, jf.FileBaseName(), line, newLine)
			}
		}
	}

	if dirty {
		return strings.Join(lines, "\n"), true
	}

	return "", false
}

// 目前还不支持多行场景
func (m *ImportResourceManager) processImportResource(line, componentName string) (string, bool) {
	return code.P.ImportResourcePattern.ReplaceAllStringFunc(line, func(match string) string {
		// 提取所有资源路径
		resourcePaths := regexp.MustCompile(`"([^"]+)"|'([^']+)'`).FindAllStringSubmatch(match, -1)

		newPaths := make([]string, 0, len(resourcePaths))
		oldResourcePaths := make([]string, 0, len(resourcePaths))
		for _, path := range resourcePaths {
			resourcePath := path[1]
			if resourcePath == "" {
				resourcePath = path[2] // 如果使用单引号
			}

			oldResourcePaths = append(oldResourcePaths, resourcePath)

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
		hasLocations := strings.Contains(match, "locations")
		if len(newPaths) == 1 && !hasLocations {
			transformer.Get().TransformImportResource(componentName, strings.Join(oldResourcePaths, ", "), newPaths[0])
			return fmt.Sprintf(`@ImportResource(%s)`, newPaths[0])
		} else {
			newValue := strings.Join(newPaths, ", ")
			transformer.Get().TransformImportResource(componentName, strings.Join(oldResourcePaths, ", "), newValue)
			return fmt.Sprintf(`@ImportResource(locations = {%s})`, newValue)
		}
	}), true
}
