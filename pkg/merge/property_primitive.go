package merge

type Key string

func (k Key) WithNamespace(ns string) string {
	return k.NamespacePrefix(ns) + string(k)
}

func (k Key) NamespacePrefix(ns string) string {
	return ns + "."
}
