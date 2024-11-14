package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
)

// 处理 @Resource 的 Bean 注入：基本操作是替换为 @Autowired，如果一个类里同一个类型有多次注入则增加 @Qualifier
type SpringBeanInjectionManager struct {
}

type ReconcileResourceToAutowiredResult struct {
	Updated int
}

func NewSpringBeanInjectionManager() *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{}
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

		javaFile := NewJavaFile(path, &component, string(fileContent))
		javaFile.format()
		newfileContent := m.reconcileJavaFile(javaFile)

		if newfileContent != javaFile.Content() { // TODO 不能以此为准了
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

func (m *SpringBeanInjectionManager) reconcileJavaFile(jf *JavaFile) string {
	// 首先应用 Bean 转换
	fileContent := jf.ApplyBeanTransformRule(jf.c.Transform.Beans)

	// 然后应用 @Resource 到 @Autowired 的转换
	return m.replaceResourceWithAutowired(jf, fileContent)
}

// replaceResourceWithAutowired 自动处理 Java 源代码，将 @Resource 注解替换为 @Autowired，
// 并在必要时添加 @Qualifier 注解。此方法还管理相关的导入语句。
func (m *SpringBeanInjectionManager) replaceResourceWithAutowired(jf *JavaFile, fileContent string) string {
	if !P.resourcePattern.MatchString(fileContent) && !P.autowiredPattern.MatchString(fileContent) {
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

	codeLines, needAutowired, needQualifier := m.processNonCommentCodeLines(jf, codeLines)
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

func (m *SpringBeanInjectionManager) processNonCommentCodeLines(jf *JavaFile, codeLines []string) (processedLines []string, needAutowired bool, needQualifier bool) {
	jc := NewJavaLines(codeLines)
	beanTypeCount := jc.InjectedBeanTypeCounts()

	for i := 0; i < len(codeLines); i++ {
		line := codeLines[i]

		if P.resourcePattern.MatchString(line) || P.autowiredPattern.MatchString(line) {
			if i+1 < len(codeLines) {
				nextLine := codeLines[i+1]
				indent := strings.TrimSuffix(line, strings.TrimSpace(line))

				if P.methodResourcePattern.MatchString(line + "\n" + nextLine) {
					// 处理方法注入
					beanType := jc.getBeanTypeFromMethodSignature(nextLine)
					processedLines = append(processedLines, indent+"@Autowired")
					needAutowired = true
					qualifierName := jc.getQualifierNameFromMethod(line, nextLine)
					if qualifierName == "" {
						// fallback
						qualifierName = strings.ToLower(beanType[:1]) + beanType[1:]
					}

					if beanTypeCount[beanType] > 1 || P.resourceWithNamePattern.MatchString(line) {
						processedLines = append(processedLines, indent+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
						needQualifier = true
					}

					processedLines = append(processedLines, nextLine)
					i++ // 跳过下一行
				} else {
					// 处理字段注入
					beanType, fieldName := jc.parseFieldDeclaration(nextLine)

					if m.shouldKeepResource(beanTypeCount, beanType, fieldName) {
						log.Printf("[%s] %s Keep @Resource for %s:%s", jf.ComponentName(), jf.FileBaseName(), beanType, fieldName)
						processedLines = append(processedLines, line, nextLine)
						i++
						continue
					}

					// 检查是否为 Map, HashMap 或 List 类型
					if P.genericTypePattern.MatchString(nextLine) {
						processedLines = append(processedLines, line, nextLine)
						i++
						continue
					}

					if P.resourcePattern.MatchString(line) {
						processedLines = append(processedLines, indent+"@Autowired")
						needAutowired = true
					} else {
						processedLines = append(processedLines, line)
					}

					if beanTypeCount[beanType] > 1 || P.resourceWithNamePattern.MatchString(line) {
						qualifierName := fieldName
						if matches := P.resourceWithNamePattern.FindStringSubmatch(line); len(matches) > 1 {
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

		// 检查是否存在 "Impl" 后缀的变体或去掉 "Impl" 后缀的变体（忽略大小写）
		if strings.HasSuffix(lowerCountFieldName, "impl") {
			if strings.TrimSuffix(lowerCountFieldName, "impl") == strings.TrimSuffix(lowerFieldName, "impl") {
				return true
			}
		} else if strings.HasSuffix(lowerFieldName, "impl") {
			if strings.TrimSuffix(lowerFieldName, "impl") == lowerCountFieldName {
				return true
			}
		} else {
			// 检查是否存在带 "Impl" 后缀的字段
			if _, exists := beanTypeCount[countFieldName+"Impl"]; exists {
				return true
			}
			implLower := strings.ToLower(countFieldName + "Impl")
			for otherFieldName := range beanTypeCount {
				if strings.ToLower(otherFieldName) == implLower {
					return true
				}
			}
		}
	}

	return false
}
