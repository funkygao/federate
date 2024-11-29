package merge

import (
	"context"
	"fmt"
	"log"
	"strings"

	"federate/pkg/code"
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

func (m *SpringBeanInjectionManager) Name() string {
	return "Reconciling Spring Bean Injection conflicts by Rewriting @Resource"
}

func (m *SpringBeanInjectionManager) Reconcile() error {
	for _, component := range m.m.Components {
		if err := m.reconcileComponent(component); err != nil {
			return err
		}
	}
	return nil
}

func (m *SpringBeanInjectionManager) reconcileComponent(component manifest.ComponentInfo) error {
	return code.NewComponentJavaWalker(component).
		AddVisitor(m).
		Walk()
}

func (m *SpringBeanInjectionManager) Visit(ctx context.Context, jf *code.JavaFile) {
	if newfileContent, dirty := m.reconcileJavaFile(jf); dirty {
		if err := jf.Overwrite(newfileContent); err != nil {
			log.Fatalf("%v", err)
		}
	}
}

func (m *SpringBeanInjectionManager) reconcileJavaFile(jf *code.JavaFile) (string, bool) {
	// 首先应用基于人工规则的注入转换
	fileContent := jf.ApplyBeanTransformRule()

	jf.UpdateContent(fileContent)

	// 然后应用 @Resource 到 @Autowired 的转换
	return m.reconcileInjectionAnnotations(jf)
}

// reconcileInjectionAnnotations 自动处理 Java 源代码，将 @Resource 注解替换为 @Autowired，
// 并在必要时添加 @Qualifier 注解。此方法还管理相关的导入语句。
func (m *SpringBeanInjectionManager) reconcileInjectionAnnotations(jf *code.JavaFile) (string, bool) {
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

func (m *SpringBeanInjectionManager) transformInjectionAnnotations(jf *code.JavaFile, bodyLines []string) (processedLines []string,
	needAutowired bool, needQualifier bool) {
	// pass 1: scan
	jl := code.NewJavaLines(bodyLines)
	beans := jl.ScanInjectedBeans()

	commentTracker := code.NewCommentTracker()

	// pass 2: transform code in place
	for i := 0; i < len(bodyLines); i++ {
		line := bodyLines[i]
		if jl.IsEmptyLine(line) || commentTracker.InComment(line) || !code.IsInjectionAnnotatedLine(line) {
			processedLines = append(processedLines, line)
			continue
		}

		if i >= len(bodyLines)-1 {
			// EOF
			return
		}

		nextLine := bodyLines[i+1]
		leadingSpace := strings.TrimSuffix(line, strings.TrimSpace(line))

		if code.IsMethodResourceAnnotatedLines(line + "\n" + nextLine) {
			// 处理方法注入
			beanType := jl.GetBeanTypeFromMethodSignature(nextLine)
			processedLines = append(processedLines, leadingSpace+"@Autowired")
			m.AutowiredN++
			needAutowired = true
			qualifierName := jl.GetQualifierNameFromMethod(line, nextLine)
			if len(beans[beanType]) > 1 || code.IsResourceAnnotatedWithNameLine(line) {
				processedLines = append(processedLines, leadingSpace+fmt.Sprintf("@Qualifier(\"%s\")", qualifierName))
				m.QualifierN++
				needQualifier = true
			}

			processedLines = append(processedLines, nextLine)
			i++ // 跳过下一行
		} else {
			// 处理字段注入
			beanType, fieldName := jl.ParseFieldDeclaration(nextLine)

			if jf.ShouldKeepResource(beans, beanType, fieldName) {
				log.Printf("[%s] %s Keep @Resource for %s %s", jf.ComponentName(), jf.FileBaseName(), beanType, fieldName)
				processedLines = append(processedLines, line, nextLine)
				i++
				continue
			}

			// 检查是否为 Map, HashMap 或 List 类型
			if code.IsCollectionTypeLine(nextLine) {
				processedLines = append(processedLines, line, nextLine)
				i++
				continue
			}

			if code.IsResourceAnnotatedLine(line) {
				processedLines = append(processedLines, leadingSpace+"@Autowired")
				m.AutowiredN++
				needAutowired = true
			} else {
				processedLines = append(processedLines, line)
			}

			if len(beans[beanType]) > 1 || code.IsResourceAnnotatedWithNameLine(line) {
				qualifierName := fieldName
				if name := code.GetResourceAnnotationName(line); name != "" {
					qualifierName = name
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

	ct := code.NewCommentTracker()
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
