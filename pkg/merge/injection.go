package merge

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

type SpringBeanInjectionManager struct {
	resourcePattern         *regexp.Regexp
	resourceWithNamePattern *regexp.Regexp
	methodResourcePattern   *regexp.Regexp
	genericTypePattern      *regexp.Regexp
	autowiredPattern        *regexp.Regexp
	qualifierPattern        *regexp.Regexp
}

type ReconcileResourceToAutowiredResult struct {
	Updated int
}

func NewSpringBeanInjectionManager() *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{
		resourcePattern:         regexp.MustCompile(`@Resource(\s*\([^)]*\))?`),
		resourceWithNamePattern: regexp.MustCompile(`@Resource\s*\(\s*name\s*=\s*"([^"]*)"\s*\)`),
		methodResourcePattern:   regexp.MustCompile(`@Resource(\s*\([^)]*\))?\s*\n\s*public\s+void\s+(set\w+)\s*\(`),
		genericTypePattern:      regexp.MustCompile(`(Map|List)<.*>`),
		autowiredPattern:        regexp.MustCompile(`@Autowired(\s*\([^)]*\))?`),
		qualifierPattern:        regexp.MustCompile(`@Qualifier\s*\(\s*"([^"]*)"\s*\)`),
	}
}

func (m *SpringBeanInjectionManager) Reconcile(manifest *manifest.Manifest, dryRun bool) (ReconcileResourceToAutowiredResult, error) {
	result := ReconcileResourceToAutowiredResult{}
	for _, component := range manifest.Components {
		updated, err := m.reconcileComponent(component, dryRun)
		if err != nil {
			return result, err
		}
		result.Updated += updated
	}
	return result, nil
}

func (m *SpringBeanInjectionManager) reconcileComponent(component manifest.ComponentInfo, dryRun bool) (int, error) {
	updated := 0
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

		oldfileContent := string(fileContent)
		oldfileContent = strings.ReplaceAll(oldfileContent, "\r\n", "\n") // 处理 Windows 风格的行尾
		oldfileContent = strings.ReplaceAll(oldfileContent, "\r", "\n")   // 处理旧 Mac 风格的行尾

		newfileContent := m.reconcileJavaFile(component, oldfileContent)

		if newfileContent != oldfileContent { // TODO 不能以此为准了
			if !dryRun {
				err = ioutil.WriteFile(path, []byte(newfileContent), info.Mode())
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

func (m *SpringBeanInjectionManager) reconcileJavaFile(component manifest.ComponentInfo, fileContent string) string {
	// 首先应用 Bean 转换
	fileContent = m.applyBeanTransforms(fileContent, component.Transform.Beans)

	// 然后应用 @Resource 到 @Autowired 的转换
	return m.replaceResourceWithAutowired(fileContent)
}

func (m *SpringBeanInjectionManager) applyBeanTransforms(fileContent string, beanTransforms map[string]string) string {
	for oldBean, newBean := range beanTransforms {
		// 更新 @Autowired 注解
		fileContent = m.autowiredPattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
		// 更新 @Qualifier 注解
		fileContent = m.qualifierPattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
		// 更新 @Resource 注解
		fileContent = m.resourceWithNamePattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
	}
	return fileContent
}

// replaceResourceWithAutowired 处理 Java 源代码，将 @Resource 注解替换为 @Autowired，
// 并在必要时添加 @Qualifier 注解。此方法还管理相关的导入语句。
func (m *SpringBeanInjectionManager) replaceResourceWithAutowired(fileContent string) string {
	if !m.resourcePattern.MatchString(fileContent) && !m.autowiredPattern.MatchString(fileContent) {
		// No changes needed
		return fileContent
	}

	var (
		packageLine string
		imports     []string
		codeLines   []string

		lines              = strings.Split(fileContent, "\n")
		inMultiLineComment = false
	)

	// 分离包声明、导入语句和代码，同时处理注释
	for _, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		if trimmedLine == "" {
			continue
		}

		if m.isInComment(line, &inMultiLineComment) {
			// 完全跳过注释行，不做任何处理
			continue
		} else if strings.HasPrefix(trimmedLine, "package ") {
			packageLine = line
		} else if strings.HasPrefix(trimmedLine, "import ") {
			imports = append(imports, line)
		} else {
			codeLines = append(codeLines, line)
		}
	}

	if len(codeLines) == 0 {
		// 该文件全是注释
		return fileContent
	}

	codeLines, needAutowired, needQualifier := m.processNonCommentCodeLines(codeLines)
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

func (m *SpringBeanInjectionManager) scanBeanTypeCounts(codeLines []string) map[string]int {
	beanTypeCount := make(map[string]int)
	for i := 0; i < len(codeLines); i++ {
		line := codeLines[i]
		if m.resourcePattern.MatchString(line) || m.autowiredPattern.MatchString(line) {
			if i+1 < len(codeLines) {
				nextLine := codeLines[i+1]
				if m.methodResourcePattern.MatchString(line + "\n" + nextLine) {
					// 方法注入
					beanType := m.getBeanTypeFromMethodSignature(nextLine)
					if beanType != "" {
						beanTypeCount[beanType]++
					}
					i++ // 跳过下一行
				} else {
					// 字段注入
					beanType, _ := m.parseFieldDeclaration(nextLine)
					if beanType != "" && !m.genericTypePattern.MatchString(nextLine) {
						beanTypeCount[beanType]++
					}
				}
			}
		}
	}
	return beanTypeCount
}

func (m *SpringBeanInjectionManager) getBeanTypeFromMethodSignature(line string) string {
	// 从方法签名中提取参数类型
	// 例如：从 "public void setService(SomeService service)" 提取 "SomeService"
	parts := strings.Split(strings.TrimSpace(line), "(")
	if len(parts) > 1 {
		paramParts := strings.Split(parts[1], ")")
		if len(paramParts) > 0 {
			typeParts := strings.Fields(paramParts[0])
			if len(typeParts) > 0 {
				return typeParts[0]
			}
		}
	}
	return ""
}

func (m *SpringBeanInjectionManager) processNonCommentCodeLines(codeLines []string) (processedLines []string, needAutowired bool, needQualifier bool) {
	beanTypeCount := m.scanBeanTypeCounts(codeLines)

	for i := 0; i < len(codeLines); i++ {
		line := codeLines[i]

		if m.resourcePattern.MatchString(line) || m.autowiredPattern.MatchString(line) {
			if i+1 < len(codeLines) {
				nextLine := codeLines[i+1]
				indent := strings.TrimSuffix(line, strings.TrimSpace(line))

				if m.methodResourcePattern.MatchString(line + "\n" + nextLine) {
					// 处理方法注入
					beanType := m.getBeanTypeFromMethodSignature(nextLine)
					processedLines = append(processedLines, indent+"@Autowired")
					needAutowired = true
					qualifierName := m.getQualifierNameFromMethod(line, nextLine)
					if qualifierName == "" {
						// fallback
						qualifierName = strings.ToLower(beanType[:1]) + beanType[1:]
					}

					if beanTypeCount[beanType] > 1 || m.resourceWithNamePattern.MatchString(line) {
						processedLines = append(processedLines, indent+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
						needQualifier = true
					}

					processedLines = append(processedLines, nextLine)
					i++ // 跳过下一行
				} else {
					// 处理字段注入
					beanType, fieldName := m.parseFieldDeclaration(nextLine)

					if m.shouldKeepResource(beanTypeCount, beanType, fieldName) {
						// 保持 @Resource 不变
						processedLines = append(processedLines, line, nextLine)
						i++
						continue
					}

					// 检查是否为 Map, HashMap 或 List 类型
					if m.genericTypePattern.MatchString(nextLine) {
						processedLines = append(processedLines, line, nextLine)
						i++
						continue
					}

					if m.resourcePattern.MatchString(line) {
						processedLines = append(processedLines, indent+"@Autowired")
						needAutowired = true
					} else {
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
					i++ // 跳过下一行
				}
			}
		} else {
			processedLines = append(processedLines, line)
		}
	}

	return processedLines, needAutowired, needQualifier
}

func (m *SpringBeanInjectionManager) getQualifierNameFromMethod(resourceLine, methodLine string) string {
	// 首先检查是否在 @Resource 中明确指定了 name
	if matches := m.resourceWithNamePattern.FindStringSubmatch(resourceLine); len(matches) > 1 {
		return matches[1]
	}

	// 如果没有明确指定 name，则从 setter 方法名中提取
	methodParts := strings.Fields(methodLine)
	for _, part := range methodParts {
		if strings.HasPrefix(part, "set") && strings.Contains(part, "(") {
			methodName := strings.Split(part, "(")[0]
			if len(methodName) > 3 && methodName[:3] == "set" && unicode.IsUpper(rune(methodName[3])) {
				// 只处理标准的 setter 方法（set后面紧跟大写字母）
				return strings.ToLower(methodName[3:4]) + methodName[4:]
			}
			// 如果不是标准的 setter 方法，直接返回空字符串
			return ""
		}
	}

	return "" // 如果无法提取到合适的名称，返回空字符串
}

func (m *SpringBeanInjectionManager) isInComment(line string, inMultiLineComment *bool) bool {
	trimmedLine := strings.TrimSpace(line)
	if strings.HasPrefix(trimmedLine, "/*") {
		*inMultiLineComment = true
	}
	if strings.HasSuffix(trimmedLine, "*/") {
		*inMultiLineComment = false
		return true // 这一行仍然是注释的一部分
	}
	return *inMultiLineComment || strings.HasPrefix(trimmedLine, "//")
}

func (m *SpringBeanInjectionManager) parseFieldDeclaration(line string) (beanType string, fieldName string) {
	// 移除行首尾的空白字符
	line = strings.TrimSpace(line)

	// 如果行以 "public void set" 开头，则跳过（这是方法，不是字段）
	if strings.HasPrefix(strings.TrimSpace(line), "public void set") {
		return "", ""
	}

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

// 由于原有Java代码书写不规范，一个接口只有一个实现类，却被注入多次。这时，不修改原有的 Resource
//
//	public class Foo {
//	    @Resource
//	    private EggService eggService;
//	    @Resource
//	    private EggService eggServiceImpl;
//	    @Resource
//	    private EggService eggserviceImpl;
//	}
func (m *SpringBeanInjectionManager) shouldKeepResource(beanTypeCount map[string]int, beanType string, fieldName string) bool {
	if beanTypeCount[beanType] <= 1 {
		return false
	}

	lowerFieldName := strings.ToLower(fieldName)

	// 检查是否存在匹配的字段名（忽略大小写）
	for countFieldName := range beanTypeCount {
		lowerCountFieldName := strings.ToLower(countFieldName)
		if lowerCountFieldName == lowerFieldName {
			continue // 跳过自身
		}

		// 检查是否存在 "Impl" 后缀的变体（忽略大小写）
		if strings.HasSuffix(lowerCountFieldName, "impl") {
			if strings.TrimSuffix(lowerCountFieldName, "impl") == strings.TrimSuffix(lowerFieldName, "impl") {
				return true
			}
		} else if strings.HasSuffix(lowerFieldName, "impl") {
			if strings.TrimSuffix(lowerFieldName, "impl") == lowerCountFieldName {
				return true
			}
		}
	}

	return false
}
