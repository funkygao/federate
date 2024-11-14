package merge

import (
	"strings"
	"unicode"
)

type JavaLines struct {
	lines []string
}

func newJavaLines(lines []string) *JavaLines {
	return &JavaLines{lines: lines}
}

// 扫描全部注入的 bean
// beans: key is bean type, value is list of field name
func (jc *JavaLines) InjectedBeans() (beans map[string][]string) {
	beans = make(map[string][]string)
	for i := 0; i < len(jc.lines); i++ {
		line := jc.lines[i]

		// 通过 @Resource/@Autowired 注入，注入可能在 setter 方法，也可能在字段
		if P.resourcePattern.MatchString(line) || P.autowiredPattern.MatchString(line) {
			if i+1 < len(jc.lines) {
				nextLine := jc.lines[i+1]
				annotatedDeclaration := line + "\n" + nextLine

				var beanType, beanName string

				switch {
				case P.methodResourcePattern.MatchString(annotatedDeclaration) ||
					P.methodAutowiredPattern.MatchString(annotatedDeclaration):
					// 处理方法注入
					beanType = jc.getBeanTypeFromMethodSignature(nextLine)
					beanName = jc.getQualifierNameFromMethod(line, nextLine)

					i++ // 跳过下一行

				default:
					// 处理字段注入
					beanType, beanName = jc.parseFieldDeclaration(nextLine)
				}

				if beanType != "" && beanName != "" && !P.genericTypePattern.MatchString(nextLine) {
					beans[beanType] = append(beans[beanType], beanName)
				}
			}
		}
	}
	return
}

func (jc *JavaLines) getBeanTypeFromMethodSignature(line string) string {
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

// @Resource
// public void setMyService(SomeService service)
//
// getQualifierNameFromMethod 返回 myService
func (jc *JavaLines) getQualifierNameFromMethod(resourceLine, methodLine string) string {
	// 首先检查是否在 @Resource 中明确指定了 name
	if matches := P.resourceWithNamePattern.FindStringSubmatch(resourceLine); len(matches) > 1 {
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

func (jc *JavaLines) parseFieldDeclaration(line string) (beanType string, fieldName string) {
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
