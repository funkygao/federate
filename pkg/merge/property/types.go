package property

import (
	"path/filepath"
	"strings"
)

type Key string

func (k Key) WithNamespace(ns string) string {
	return ns + "." + string(k)
}

type PropertySource struct {
	Value          interface{}
	OriginalString string
	FilePath       string
}

func (ps *PropertySource) IsYAML() bool {
	ext := ps.fileExt()
	return ext == ".yaml" || ext == ".yml"
}

func (ps *PropertySource) fileExt() string {
	return strings.ToLower(filepath.Ext(ps.FilePath))
}

func (ps *PropertySource) IsProperties() bool {
	ext := ps.fileExt()
	return ext == ".properties"
}
