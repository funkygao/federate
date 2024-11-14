package merge

import (
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

type JavaFile struct {
	c *manifest.ComponentInfo

	path    string
	content string
}

func NewJavaFile(path string, c *manifest.ComponentInfo, content string) *JavaFile {
	// reformat
	content = strings.ReplaceAll(content, "\r\n", "\n") // 处理 Windows 风格的行尾
	content = strings.ReplaceAll(content, "\r", "\n")   // 处理旧 Mac 风格的行尾

	return &JavaFile{
		c:       c,
		path:    path,
		content: content,
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
}

func (j *JavaFile) Content() string {
	return j.content
}

func (j *JavaFile) JavaLines() *JavaLines {
	return NewJavaLines(j.lines())
}

func (j *JavaFile) lines() []string {
	return strings.Split(j.content, "\n")
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
