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
	m *manifest.Manifest

	AutowiredN int
	QualifierN int
}

func NewSpringBeanInjectionManager(m *manifest.Manifest) *SpringBeanInjectionManager {
	return &SpringBeanInjectionManager{m: m}
}

func (m *SpringBeanInjectionManager) Reconcile(dryRun bool) error {
	for _, component := range m.m.Components {
		if err := m.reconcileComponent(component, dryRun); err != nil {
			return err
		}
	}
	return nil
}

func (m *SpringBeanInjectionManager) reconcileComponent(component manifest.ComponentInfo, dryRun bool) error {
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
		newfileContent, dirty := m.reconcileJavaFile(javaFile)

		if !dryRun && dirty {
			err = ioutil.WriteFile(path, []byte(newfileContent), info.Mode())
			if err != nil {
				return err
			}
		}
		return nil
	})
	return err
}

func (m *SpringBeanInjectionManager) reconcileJavaFile(jf *JavaFile) (string, bool) {
	// 首先应用基于人工规则的注入转换
	fileContent := jf.ApplyBeanTransformRule(jf.c.Transform.Beans)

	jf.UpdateContent(fileContent)

	// 然后应用 @Resource 到 @Autowired 的转换
	return m.reconcileInjectionAnnotations(jf)
}

// reconcileInjectionAnnotations 自动处理 Java 源代码，将 @Resource 注解替换为 @Autowired，
// 并在必要时添加 @Qualifier 注解。此方法还管理相关的导入语句。
func (m *SpringBeanInjectionManager) reconcileInjectionAnnotations(jf *JavaFile) (string, bool) {
	if !jf.HasInjectionAnnotation() {
		return jf.Content(), false
	}

	jl := jf.JavaLines()
	jl.SeparateSections()
	if jl.EmptyCode() {
		return jf.Content(), false
	}

	bodyLines, needAutowired, needQualifier := m.transformInjectionAnnotations(jf, jl.BodyLines())
	result := m.transformImportIfNec(jl.HeadLines(), needAutowired, needQualifier)
	result = append(result, bodyLines...)
	return strings.Join(result, "\n"), needAutowired || needQualifier
}

func (m *SpringBeanInjectionManager) transformInjectionAnnotations(jf *JavaFile, bodyLines []string) (processedLines []string,
	needAutowired bool, needQualifier bool) {
	// pass 1: scan
	jc := newJavaLines(bodyLines)
	beans := jc.ScanInjectedBeans()

	commentTracker := NewCommentTracker()

	// pass 2: transform code in place
	for i := 0; i < len(bodyLines); i++ {
		line := bodyLines[i]
		if jc.IsEmptyLine(line) || commentTracker.InComment(line) || !P.IsInjectionAnnotatedLine(line) {
			processedLines = append(processedLines, line)
			continue
		}

		if i >= len(bodyLines)-1 {
			// EOF
			return
		}

		nextLine := bodyLines[i+1]
		leadingSpace := strings.TrimSuffix(line, strings.TrimSpace(line))

		if P.methodResourcePattern.MatchString(line + "\n" + nextLine) {
			// 处理方法注入
			beanType := jc.getBeanTypeFromMethodSignature(nextLine)
			processedLines = append(processedLines, leadingSpace+"@Autowired")
			m.AutowiredN++
			needAutowired = true
			qualifierName := jc.getQualifierNameFromMethod(line, nextLine)
			if len(beans[beanType]) > 1 || P.resourceWithNamePattern.MatchString(line) {
				processedLines = append(processedLines, leadingSpace+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
				m.QualifierN++
				needQualifier = true
			}

			processedLines = append(processedLines, nextLine)
			i++ // 跳过下一行
		} else {
			// 处理字段注入
			beanType, fieldName := jc.parseFieldDeclaration(nextLine)

			if jf.shouldKeepResource(beans, beanType, fieldName) {
				log.Printf("[%s] %s Keep @Resource for %s %s", jf.ComponentName(), jf.FileBaseName(), beanType, fieldName)
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
				processedLines = append(processedLines, leadingSpace+"@Autowired")
				m.AutowiredN++
				needAutowired = true
			} else {
				processedLines = append(processedLines, line)
			}

			if len(beans[beanType]) > 1 || P.resourceWithNamePattern.MatchString(line) {
				qualifierName := fieldName
				if matches := P.resourceWithNamePattern.FindStringSubmatch(line); len(matches) > 1 {
					qualifierName = matches[1]
				}
				processedLines = append(processedLines, leadingSpace+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
				m.QualifierN++

				needQualifier = true
			}

			processedLines = append(processedLines, nextLine)
			i++ // 跳过下一行
		}
	}

	return processedLines, needAutowired, needQualifier
}

func (m *SpringBeanInjectionManager) transformImportIfNec(headLines []string, needAutowired, needQualifier bool) []string {
	var processedHeadLines []string
	autowiredImported := false
	qualifierImported := false

	ct := NewCommentTracker()
	for _, line := range headLines {
		processedHeadLines = append(processedHeadLines, line)
		if ct.InComment(line) {
			continue
		}

		switch {
		case strings.Contains(line, "org.springframework.beans.factory.annotation.Autowired"):
			autowiredImported = true
		case strings.Contains(line, "org.springframework.beans.factory.annotation.Qualifier"):
			qualifierImported = true
		}
	}

	// Append imports if neccessary
	if needAutowired && !autowiredImported {
		processedHeadLines = append(processedHeadLines, "import org.springframework.beans.factory.annotation.Autowired;")
	}
	if needQualifier && !qualifierImported {
		processedHeadLines = append(processedHeadLines, "import org.springframework.beans.factory.annotation.Qualifier;")
	}

	return processedHeadLines
}
