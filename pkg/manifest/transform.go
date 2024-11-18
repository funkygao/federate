package manifest

type TransformSpec struct {
	Beans     map[string]string `yaml:"beans"`
	Autowired AutowiredSpec     `yaml:"autowired"`
	Services  map[string]string `yaml:"service"`
}

type AutowiredSpec struct {
	Excludes []string `yaml:"exclude"`
}

func (a *AutowiredSpec) ExcludeBeanType(beanType string) bool {
	for _, excluded := range a.Excludes {
		if beanType == excluded {
			return true
		}
	}
	return false
}
