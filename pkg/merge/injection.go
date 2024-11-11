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
	autowiredPattern        *regexp.Regexp
}

type ReconcileResourceToAutowiredResult struct {
	Updated int
}

func NewSpringBeanInjectionManager() *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{
		resourcePattern:         regexp.MustCompile(`@Resource(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		genericTypePattern:      regexp.MustCompile(`(Map|List)<.*>`),
		autowiredPattern:        regexp.MustCompile(`@Autowired(\s*\([^)]*\))?`),
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
	if !m.resourcePattern.MatchString(content) && !m.autowiredPattern.MatchString(content) {
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
	resourceImported := false

	for _, imp := range imports {
		switch {
		case strings.Contains(imp, "org.springframework.beans.factory.annotation.Autowired"):
			autowiredImported = true
		case strings.Contains(imp, "org.springframework.beans.factory.annotation.Qualifier"):
			qualifierImported = true
		case strings.Contains(imp, "javax.annotation.Resource"):
			resourceImported = true
		}
		processedImports = append(processedImports, imp)
	}

	if needAutowired && !autowiredImported {
		processedImports = append(processedImports, "import org.springframework.beans.factory.annotation.Autowired;")
	}
	if needQualifier && !qualifierImported {
		processedImports = append(processedImports, "import org.springframework.beans.factory.annotation.Qualifier;")
	}
	if !resourceImported {
		// Remove Resource import if it's not needed anymore
		processedImports = removeResourceImport(processedImports)
	}

	return processedImports
}

func removeResourceImport(imports []string) []string {
	var result []string
	for _, imp := range imports {
		if !strings.Contains(imp, "javax.annotation.Resource") {
			result = append(result, imp)
		}
	}
	return result
}

func (m *SpringBeanInjectionManager) processCodeLines(codeLines []string) (processedLines []string, needAutowired bool, needQualifier bool) {
	beanTypeCount := make(map[string]int)
	beanLines := make(map[string][]int)
	inMultiLineComment := false

	// 辅助函数：检查是否在注释中
	isInComment := func(line string) bool {
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, "/*") {
			inMultiLineComment = true
		}
		if strings.HasSuffix(trimmedLine, "*/") {
			inMultiLineComment = false
		}
		return inMultiLineComment || strings.HasPrefix(trimmedLine, "//")
	}

	// 第一次扫描：计数并记录行号
	for i, line := range codeLines {
		if isInComment(line) {
			continue
		}
		if m.resourcePattern.MatchString(line) || m.autowiredPattern.MatchString(line) {
			if i+1 < len(codeLines) {
				beanType, _ := m.extractBeanInfo(codeLines[i+1])
				if beanType != "" && !m.genericTypePattern.MatchString(codeLines[i+1]) {
					beanTypeCount[beanType]++
					beanLines[beanType] = append(beanLines[beanType], i)
				}
			}
		}
	}

	// 重置多行注释标志
	inMultiLineComment = false

	// 第二次扫描：进行实际替换
	for i := 0; i < len(codeLines); i++ {
		line := codeLines[i]

		if isInComment(line) {
			processedLines = append(processedLines, line)
			continue
		}

		if m.resourcePattern.MatchString(line) || m.autowiredPattern.MatchString(line) {
			if i+1 < len(codeLines) && m.genericTypePattern.MatchString(codeLines[i+1]) {
				// 保持 Map 和 List 的 @Resource 不变
				processedLines = append(processedLines, line, codeLines[i+1])
				i++
			} else if i+1 < len(codeLines) {
				indent := strings.TrimSuffix(line, strings.TrimSpace(line))
				nextLine := codeLines[i+1]
				beanType, fieldName := m.extractBeanInfo(nextLine)

				if m.resourcePattern.MatchString(line) {
					// 替换 @Resource 为 @Autowired
					processedLines = append(processedLines, indent+"@Autowired")
					needAutowired = true
				} else {
					// 保持原有的 @Autowired 注解
					processedLines = append(processedLines, line)
				}

				if beanTypeCount[beanType] > 1 || m.resourceWithNamePattern.MatchString(line) {
					qualifierName := fieldName
					if matches := m.resourceWithNamePattern.FindStringSubmatch(line); len(matches) > 1 {
						qualifierName = matches[1]
					}
					processedLines = append(processedLines, indent+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
					needQualifier = true
				}

				processedLines = append(processedLines, nextLine)
				i++ // 跳过下一行，因为我们已经处理了它
			}
		} else {
			processedLines = append(processedLines, line)
		}
	}
	return
}

func (m *SpringBeanInjectionManager) extractBeanInfo(line string) (beanType string, fieldName string) {
	// 移除行首尾的空白字符
	line = strings.TrimSpace(line)

	// 处理泛型
	genericDepth := 0
	var typeBuilder strings.Builder
	var nameBuilder strings.Builder
	inType := false

	words := strings.Fields(line)
	startIndex := 0

	// 跳过访问修饰符
	if len(words) > 0 && (words[0] == "private" || words[0] == "public" || words[0] == "protected") {
		startIndex = 1
	}

	for i := startIndex; i < len(words); i++ {
		word := words[i]
		for _, char := range word {
			if char == '<' {
				genericDepth++
			} else if char == '>' {
				genericDepth--
			}

			if !inType {
				typeBuilder.WriteRune(char)
			} else if char != ';' && char != '=' {
				nameBuilder.WriteRune(char)
			}
		}

		if genericDepth == 0 {
			if !inType {
				inType = true
			} else {
				break
			}
		}

		if inType && i < len(words)-1 {
			nameBuilder.WriteRune(' ')
		}
	}

	beanType = strings.TrimSpace(typeBuilder.String())
	fieldName = strings.TrimSpace(nameBuilder.String())

	// 处理无效输入
	if beanType == "" || fieldName == "" {
		return "", ""
	}

	return beanType, fieldName
}

func (m *SpringBeanInjectionManager) isInComment(lines []string, currentIndex int) bool {
	inMultiLineComment := false
	for i := 0; i <= currentIndex; i++ {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "//") {
			continue
		}
		if strings.HasPrefix(line, "/*") || strings.HasPrefix(line, "/**") {
			inMultiLineComment = true
		}
		if strings.HasSuffix(line, "*/") {
			inMultiLineComment = false
		}
	}
	return inMultiLineComment
}
