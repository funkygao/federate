package merge

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
	FileType string // "yaml" 或 "properties"
}

type PropertyReference struct {
	Component string
	Key       string
	Value     string
	IsYAML    bool
	FilePath  string
}

func (ps *PropertySource) IsYAML() bool {
	return ps.FileType == "yaml"
}
