package bean

import (
	"regexp"
)

const (
	beanIdPathSeparator      = "." // 加载所有bean到内存时，嵌套关系
	beanIdReconcileSeparator = "-" // `#` 不允许，see https://www.w3.org/TR/xmlschema-2/#ID
)

// Xxx.getBean
var getBeanPattern = regexp.MustCompile(`\bgetBean\s*\(\s*"([^"]+)"\s*\)`)
