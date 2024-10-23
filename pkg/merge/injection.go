package merge

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

type SpringBeanInjectionManager struct {
	resourcePattern         *regexp.Regexp
	resourceWithNamePattern *regexp.Regexp
	genericTypePattern      *regexp.Regexp
}

type ReconcileResourceToAutowiredResult struct {
	Updated int
}

func NewSpringBeanInjectionManager() *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{
		resourcePattern:         regexp.MustCompile(`@Resource(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		genericTypePattern:      regexp.MustCompile(`(Map|List)<.*>`),
	}
}

func (m *SpringBeanInjectionManager) ReconcileResourceToAutowired(manifest *manifest.Manifest, dryRun bool) (ReconcileResourceToAutowiredResult, error) {
	result := ReconcileResourceToAutowiredResult{}
	for _, component := range manifest.Components {
		updated, err := m.reconcileComponentInjections(component, dryRun)
		if err != nil {
			return result, err
		}
		result.Updated += updated
	}
	return result, nil
}

func (m *SpringBeanInjectionManager) reconcileComponentInjections(component manifest.ComponentInfo, dryRun bool) (int, error) {
	updated := 0
	err := filepath.Walk(component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !java.IsJavaMainSource(info, path) {
			return nil
		}

		content, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}

		oldContent := string(content)
		newContent := m.replaceResourceWithAutowired(oldContent)

		if newContent != oldContent {
			if !dryRun {
				err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
				if err != nil {
					return err
				}
				updated++
			}
		}
		return nil
	})
	return updated, err
}

// replaceResourceWithAutowired 处理 Java 源代码，将 @Resource 注解替换为 @Autowired，
// 并在必要时添加 @Qualifier 注解。此方法还管理相关的导入语句。
//
// 此方法执行以下操作：
// 1. 保持包声明在文件的最开始。
// 2. 将所有导入语句放在包声明之后，其他内容之前。
// 3. 将 @Resource 替换为 @Autowired，除非它用于 Map<> 或 List<> 类型的字段。
// 4. 如果 @Resource 有 name 参数，添加相应的 @Qualifier 注解。
// 5. 管理导入语句：
//   - 添加必要的 org.springframework.beans.factory.annotation.Autowired 导入。
//   - 如果需要，添加 org.springframework.beans.factory.annotation.Qualifier 导入。
//   - 如果仍然使用 @Resource，保留 javax.annotation.Resource 或 javax.annotation.* 导入。
//   - 移除不再需要的 javax.annotation.Resource 导入。
//
// 6. 保持其他注解（如 @PostConstruct, @PreDestroy）不变，并确保它们的导入被保留。
//
// 参数：
//
//	content: 输入的 Java 源代码字符串。
//
// 返回：
//
//	处理后的 Java 源代码字符串。
//
// 注意：
//   - 此方法假设输入的 Java 源代码格式良好。
//   - 它不会更改代码的整体结构或缩进。
//   - 对于复杂的泛型类型（如嵌套的 Map 或 List），可能需要人工审查结果。
func (m *SpringBeanInjectionManager) replaceResourceWithAutowired(content string) string {
	if !m.resourcePattern.MatchString(content) {
		// No changes needed
		return content
	}

	lines := strings.Split(content, "\n")
	var (
		packageLine string
		imports     []string
		codeLines   []string
	)

	// 分离包声明、导入语句和代码
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "package ") {
			packageLine = line
		} else if strings.HasPrefix(strings.TrimSpace(line), "import ") {
			imports = append(imports, line)
		} else {
			codeLines = append(codeLines, line)
		}
	}

	codeLines, needAutowired, needQualifier := m.processCodeLines(codeLines)
	imports = m.processImports(imports, needAutowired, needQualifier)

	result := []string{packageLine}
	result = append(result, imports...)
	result = append(result, codeLines...)
	return strings.Join(result, "\n")
}

func (m *SpringBeanInjectionManager) processImports(imports []string, needAutowired, needQualifier bool) []string {
	var processedImports []string
	autowiredImported := false
	qualifierImported := false

	for _, imp := range imports {
		switch {
		case strings.Contains(imp, "org.springframework.beans.factory.annotation.Autowired"):
			autowiredImported = true
		case strings.Contains(imp, "org.springframework.beans.factory.annotation.Qualifier"):
			qualifierImported = true
		}
		processedImports = append(processedImports, imp)
	}

	if needAutowired && !autowiredImported {
		processedImports = append(processedImports, "import org.springframework.beans.factory.annotation.Autowired;")
	}
	if needQualifier && !qualifierImported {
		processedImports = append(processedImports, "import org.springframework.beans.factory.annotation.Qualifier;")
	}

	return processedImports
}

func (m *SpringBeanInjectionManager) processCodeLines(codeLines []string) ([]string, bool, bool) {
	var processedLines []string
	needAutowired := false
	needQualifier := false

	for i, line := range codeLines {
		if m.resourcePattern.MatchString(line) {
			if i+1 < len(codeLines) && m.genericTypePattern.MatchString(codeLines[i+1]) {
				// 保持 Map 和 List 的 @Resource 不变
				processedLines = append(processedLines, line)
			} else if matches := m.resourceWithNamePattern.FindStringSubmatch(line); len(matches) > 1 {
				indent := strings.TrimSuffix(line, strings.TrimSpace(line))
				processedLines = append(processedLines,
					indent+"@Autowired",
					indent+fmt.Sprintf("@Qualifier(\"%s\")", matches[1]))
				needAutowired = true
				needQualifier = true
			} else {
				processedLines = append(processedLines, strings.Replace(line, "@Resource", "@Autowired", 1))
				needAutowired = true
			}
		} else {
			processedLines = append(processedLines, line)
		}
	}
	return processedLines, needAutowired, needQualifier
}
