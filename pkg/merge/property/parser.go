package property

import (
	"federate/pkg/manifest"
)

// PropertyParser 定义了属性文件解析器的接口
type PropertyParser interface {
	Parse(filePath string, component manifest.ComponentInfo, cm *PropertyManager) error

	Generate(entries map[string]PropertyEntry, rawKeys []string, targetFile string) error
}
