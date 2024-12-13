package property

import (
	"federate/pkg/manifest"
)

// PropertyParser 定义了属性文件解析器的接口
type PropertyParser interface {
	Parse(string, manifest.ComponentInfo, *PropertyManager) error

	Generate(entries map[string]PropertyEntry, targetFile string) error
}
