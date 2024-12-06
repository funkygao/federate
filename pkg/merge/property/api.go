package property

import (
	"fmt"
	"strings"

	"federate/pkg/manifest"
)

func (pm *PropertyManager) ContainsKey(c manifest.ComponentInfo, key string) bool {
	_, ok := pm.r.getComponentProperty(c, key)
	return ok
}

func (pm *PropertyManager) Resolve(key string) any {
	for _, entries := range pm.r.resolvableEntries {
		if entry, ok := entries[key]; ok {
			return entry.Value
		}
	}

	return nil
}

// 自动获取 line 里的属性引用占位符，并解析对应属性值，返回解析后的 line
// 如果没有占位符，则返回原 line
func (pm *PropertyManager) ResolveLine(line string) string {
	// 使用正则表达式找到所有的占位符
	matches := P.placeholderRegex.FindAllStringSubmatchIndex(line, -1)

	// 如果没有匹配项，直接返回原始行
	if len(matches) == 0 {
		return line
	}

	// 创建一个新的字符串构建器
	var result strings.Builder
	lastIndex := 0

	for _, match := range matches {
		// 添加占位符之前的文本
		result.WriteString(line[lastIndex:match[0]])

		// 提取占位符中的键
		key := line[match[2]:match[3]]

		// 解析键对应的值
		value := pm.Resolve(key)

		// 如果找到了值，添加到结果中；否则保留原始占位符
		if value != nil {
			result.WriteString(fmt.Sprintf("%v", value))
		} else {
			result.WriteString(line[match[0]:match[1]])
		}

		lastIndex = match[1]
	}

	// 添加最后一个占位符之后的文本
	result.WriteString(line[lastIndex:])

	return result.String()
}
