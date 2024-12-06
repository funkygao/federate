package property

import (
	"path/filepath"
	"strings"
)

type Key string

func (k Key) WithNamespace(ns string) string {
	return ns + "." + string(k)
}

type PropertyEntry struct {
	// 最新值：如果引用，它被替换为解析后的值
	Value any

	// 最原始值
	Raw string

	FilePath string
}

func (e *PropertyEntry) IsYAML() bool {
	ext := e.fileExt()
	return ext == ".yaml" || ext == ".yml"
}

func (e *PropertyEntry) fileExt() string {
	return strings.ToLower(filepath.Ext(e.FilePath))
}

func (e *PropertyEntry) IsProperties() bool {
	ext := e.fileExt()
	return ext == ".properties"
}

func (e *PropertyEntry) StringValue() string {
	if s, ok := e.Value.(string); ok {
		return s
	}
	return ""
}

func (e *PropertyEntry) WasReference() bool {
	return strings.Contains(e.Raw, "${")
}

func (e *PropertyEntry) RawReferenceValue() string {
	strValue := e.StringValue()
	if strValue == "" || !strings.Contains(strValue, "${") {
		return ""
	}
	return strValue
}
