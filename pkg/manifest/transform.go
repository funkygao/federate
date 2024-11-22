package manifest

import (
	"strings"

	"federate/pkg/java"
)

type TransformSpec struct {
	Beans     map[string]string `yaml:"beans"`
	Autowired AutowiredSpec     `yaml:"autowired"`
	Services  map[string]string `yaml:"service"`
	TxManager string            `yaml:"txManager"`
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

func (t *TransformSpec) ServiceBeanRefMap() map[string]string {
	r := make(map[string]string)
	for fqcn, v := range t.Services {
		simpleClassName := java.ClassSimpleName(fqcn)
		if len(simpleClassName) > 0 {
			lowercaseFirstChar := strings.ToLower(simpleClassName[:1]) + simpleClassName[1:]
			r[lowercaseFirstChar] = v
		}
	}
	return r
}
