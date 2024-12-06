package merge

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"

	"federate/pkg/code"
	"federate/pkg/diff"
	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
)

// @ImportResource
type ImportResourceManager struct {
	m *manifest.Manifest

	resourcePathPattern *regexp.Regexp

	ImportResourceCount int
}

func NewImportResourceManager(m *manifest.Manifest) Reconciler {
	return newImportResourceManager(m)
}

func newImportResourceManager(m *manifest.Manifest) *ImportResourceManager {
	return &ImportResourceManager{
		m:                   m,
		resourcePathPattern: regexp.MustCompile(`"([^"]+)"|'([^']+)'`),
	}
}

func (m *ImportResourceManager) Name() string {
	return "Transform Java @ImportResource value with '/federated/'"
}

func (m *ImportResourceManager) Reconcile() error {
	for _, component := range m.m.Components {
		if err := m.reconcileComponent(component); err != nil {
			return err
		}
	}
	return nil
}

func (m *ImportResourceManager) reconcileComponent(component manifest.ComponentInfo) error {
	return code.NewComponentJavaWalker(component).
		AddVisitor(m).
		Walk()
}

func (m *ImportResourceManager) Visit(ctx context.Context, jf *code.JavaFile) {
	if newFileContent, dirty := m.reconcileJavaFile(jf); dirty {
		diff.RenderUnifiedDiff(jf.Content(), newFileContent)

		if err := jf.Overwrite(newFileContent); err != nil {
			log.Fatalf("%v", err)
		}
	}
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
		resourcePaths := m.resourcePathPattern.FindAllStringSubmatch(match, -1)
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
			newValue := newPaths[0]
			ledger.Get().TransformImportResource(componentName, strings.Join(oldResourcePaths, ", "), strings.Trim(newValue, `"`))
			return fmt.Sprintf(`@ImportResource(%s)`, newValue)
		} else {
			newValue := strings.Join(newPaths, ", ")
			ledger.Get().TransformImportResource(componentName, strings.Join(oldResourcePaths, ", "), newValue)
			return fmt.Sprintf(`@ImportResource(locations = {%s})`, newValue)
		}
	}), true
}
