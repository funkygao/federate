package merge

import (
	"log"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

type JavaFile struct {
	c *manifest.ComponentInfo

	path    string
	content string

	cachedLines []string
}

func NewJavaFile(path string, c *manifest.ComponentInfo, content string) *JavaFile {
	// reformat
	content = strings.ReplaceAll(content, "\r\n", "\n") // 处理 Windows 风格的行尾
	content = strings.ReplaceAll(content, "\r", "\n")   // 处理旧 Mac 风格的行尾

	return &JavaFile{
		c:           c,
		path:        path,
		content:     content,
		cachedLines: nil,
	}
}

func (j *JavaFile) ComponentName() string {
	if j.c == nil {
		return "nil"
	}

	return j.c.Name
}

func (j *JavaFile) FileBaseName() string {
	return filepath.Base(j.path)
}

func (j *JavaFile) UpdateContent(content string) {
	j.content = content
	// kill cache
	j.cachedLines = nil
}

func (j *JavaFile) Content() string {
	return j.content
}

func (j *JavaFile) JavaLines() *JavaLines {
	return newJavaLines(j.lines())
}

func (j *JavaFile) lines() []string {
	if j.cachedLines == nil {
		j.cachedLines = strings.Split(j.content, "\n")
	}
	return j.cachedLines
}

func (jf *JavaFile) HasInjectionAnnotation() bool {
	// 可能会匹配到注释中的 "@Resource" 或 "@Autowired"，导致假阳性
	return P.resourcePattern.MatchString(jf.content) ||
		P.autowiredPattern.MatchString(jf.content)
}

// 根据 manifest 里人为指定的 bean 替换规则进行替换
// 例如：之前都是 DataSource dataSource，合并后有多个数据源，需要区分它们，人为指定为：
// fooDataSource, barDataSource
func (j *JavaFile) ApplyBeanTransformRule(beanTransformRule map[string]string) (fileContent string) {
	fileContent = j.content

	for oldBean, newBean := range beanTransformRule {
		// 更新 @Autowired 注解
		fileContent = P.autowiredPattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
		// 更新 @Qualifier 注解
		fileContent = P.qualifierPattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
		// 更新 @Resource 注解
		fileContent = P.resourceWithNamePattern.ReplaceAllStringFunc(fileContent, func(match string) string {
			return strings.Replace(match, oldBean, newBean, 1)
		})
	}
	return
}

// 由于原有Java代码书写不规范，一个接口只有一个实现类，却被注入多次。这时，不修改原有的 Resource
// 规则：同一类行多次注入，而且 Impl 成对出现，则保持原有注入方式，不替换
//
//	public class Foo {
//	    @Resource
//	    private EggService eggService;
//	    @Resource
//	    private EggService eggServiceImpl;
//	    @Resource
//	    private EggService eggserviceImpl;
//	}
func (j *JavaFile) shouldKeepResource(beans map[string][]string, beanType string, fieldName string) bool {
	fieldNames, exists := beans[beanType]
	if !exists || len(fieldNames) <= 1 {
		return false
	}

	if j.c != nil && j.c.Transform.Autowired.ExcludeBeanType(beanType) {
		log.Printf("[%s] Bean[%s] Excluded from Autowired Transform", j.c.Name, beanType)
		return true
	}

	lowerFieldName := strings.ToLower(fieldName)
	hasImpl := strings.HasSuffix(lowerFieldName, "impl")

	for _, otherFieldName := range fieldNames {
		if otherFieldName == fieldName {
			continue // 跳过自身
		}

		lowerOtherFieldName := strings.ToLower(otherFieldName)

		// 只有当存在配对的 Impl 和非 Impl 字段时，才保留 @Resource
		if hasImpl {
			if strings.TrimSuffix(lowerFieldName, "impl") == lowerOtherFieldName {
				return true
			}
		} else {
			if lowerFieldName == strings.TrimSuffix(lowerOtherFieldName, "impl") {
				return true
			}
		}
	}
	return false
}
