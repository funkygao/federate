package merge

import (
	"path/filepath"
	"strings"
)

type Key string

func (k Key) WithNamespace(ns string) string {
	return k.NamespacePrefix(ns) + string(k)
}

func (k Key) NamespacePrefix(ns string) string {
	return ns + "."
}

type PropertySource struct {
	Value    interface{}
	FilePath string
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

type PropertyReference struct {
	Component string
	Key       string
	Value     string
	FilePath  string
}

func (pr *PropertyReference) IsYAML() bool {
	ext := strings.ToLower(filepath.Ext(pr.FilePath))
	return ext == ".yaml" || ext == ".yml"
}
