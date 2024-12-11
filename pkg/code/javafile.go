package code

import (
	"context"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

type JavaFile struct {
	c       *manifest.ComponentInfo
	Context context.Context

	path string
	info os.FileInfo

	bytes       []byte
	content     string
	cachedLines []string

	cachedCompactCode string

	visitors []JavaFileVisitor
}

func NewJavaFile(path string, c *manifest.ComponentInfo, b []byte) *JavaFile {
	// reformat
	content := string(b)
	content = strings.ReplaceAll(content, "\r\n", "\n") // 处理 Windows 风格的行尾
	content = strings.ReplaceAll(content, "\r", "\n")   // 处理旧 Mac 风格的行尾

	return &JavaFile{
		c:           c,
		path:        path,
		bytes:       b,
		content:     content,
		cachedLines: nil,
		visitors:    []JavaFileVisitor{},
	}
}

func NewJavaFileWithContent(b []byte) *JavaFile {
	return &JavaFile{
		content: string(b),
	}
}

func (j *JavaFile) withInfo(info os.FileInfo) *JavaFile {
	j.info = info
	return j
}

func (j *JavaFile) ComponentName() string {
	if j.c == nil {
		return "nil"
	}

	return j.c.Name
}

func (j *JavaFile) Path() string {
	return j.path
}

func (j *JavaFile) FileBaseName() string {
	return filepath.Base(j.path)
}

func (j *JavaFile) CompactCode() string {
	if j.cachedCompactCode != "" {
		return j.cachedCompactCode
	}

	// Remove package declaration
	noPackage := P.packageDeclarationRegex.ReplaceAllString(j.content, "")

	// Remove comments
	noComments := P.commentRegex.ReplaceAllString(noPackage, "")

	// Remove annotations
	noAnnotations := P.annotationRegex.ReplaceAllString(noComments, "")

	// Remove whitespace
	noWhitespace := P.whitespaceRegex.ReplaceAllString(noAnnotations, "")

	// Remove semicolons
	noSemicolons := strings.ReplaceAll(noWhitespace, ";", "")

	j.cachedCompactCode = noSemicolons

	return j.cachedCompactCode
}

func (j *JavaFile) UpdateContent(content string) {
	j.content = content
	// kill cache
	j.cachedLines = nil
}

func (j *JavaFile) Overwrite(content string) error {
	return ioutil.WriteFile(j.path, []byte(content), j.info.Mode())
}

func (j *JavaFile) Bytes() []byte {
	return j.bytes
}

func (j *JavaFile) Content() string {
	return j.content
}

func (j *JavaFile) JavaLines() *JavaLines {
	return NewJavaLines(j.RawLines())
}

func (j *JavaFile) RawLines() []string {
	if j.cachedLines == nil {
		j.cachedLines = strings.Split(j.content, "\n")
	}
	return j.cachedLines
}

// AddVisitor registers a new JavaFileVisitor to be applied when Accept is called.
func (j *JavaFile) AddVisitor(visitors ...JavaFileVisitor) {
	j.visitors = append(j.visitors, visitors...)
}

// Accept applies all registered visitors to the Java file.
// It can optionally take a context via the WithContext option.
func (j *JavaFile) Accept(opts ...AcceptOption) {
	options := &acceptOptions{
		ctx: context.Background(), // Default to background context
	}
	for _, opt := range opts {
		opt(options)
	}

	for _, v := range j.visitors {
		v.Visit(options.ctx, j)
	}
}

func (jf *JavaFile) HasInjectionAnnotation() bool {
	// 可能会匹配到注释中的 "@Resource" 或 "@Autowired"，导致假阳性
	return IsInjectionAnnotatedLine(jf.content)
}

// 根据 manifest 里人为指定的 bean 替换规则进行替换
// 例如：之前都是 DataSource dataSource，合并后有多个数据源，需要区分它们，人为指定为：
// fooDataSource, barDataSource
func (j *JavaFile) ApplyBeanTransformRule() (fileContent string) {
	return j.applyBeanTransformRule(j.c.Transform.Beans)
}

func (j *JavaFile) applyBeanTransformRule(beanTransformRule map[string]string) (fileContent string) {
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
func (j *JavaFile) ShouldKeepResource(beans map[string][]string, beanType string, fieldName string) bool {
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
