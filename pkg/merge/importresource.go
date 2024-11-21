package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/diff"
	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/manifest"
)

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
		javaFile := NewJavaFile(path, &component, oldFileConent)
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
	return P.importResourcePattern.ReplaceAllStringFunc(line, func(match string) string {
		// 检查是否使用了 locations 属性
		hasLocations := strings.Contains(match, "locations")

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
		if len(newPaths) == 1 && !hasLocations {
			return fmt.Sprintf(`@ImportResource(%s)`, newPaths[0])
		} else {
			return fmt.Sprintf(`@ImportResource(locations = {%s})`, strings.Join(newPaths, ", "))
		}
	}), true
}
