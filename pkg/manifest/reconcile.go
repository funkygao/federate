package manifest

import (
	"federate/pkg/util"
)

type ReconcileSpec struct {
	ExcludedBeanClasses []string `yaml:"excludeBeans"`

	Taint     TaintSpec              `yaml:"manual"`
	Resources ResourcesReconcileSpec `yaml:"resources"`

	Rpc RpcSpec `yaml:"rpc"`

	M *MainSystem
}

type ResourcesReconcileSpec struct {
	// xml 里定义的 bean
	SingletonBeanClasses []string               `yaml:"singletonClasses"`
	FlatCopy             []string               `yaml:"copy"`
	PropertySettlement   map[string]interface{} `yaml:"override"`
}

func (s *ReconcileSpec) ExcludeBeanClass(class string) bool {
	return util.Contains(class, s.ExcludedBeanClasses)
}

func (s *ReconcileSpec) SingletonBeanClass(class string) bool {
	return util.Contains(class, s.Resources.SingletonBeanClasses)
}

func (s *ReconcileSpec) PropertySettled(key string) bool {
	_, exists := s.Resources.PropertySettlement[key]
	return exists
}

type TaintSpec struct {
	LogConfigXml     string `yaml:"logConfigXml"`
	MybatisConfigXml string `yaml:"mybatisConfigXml"`
}

func (t *TaintSpec) ResourceFiles() []string {
	return []string{t.LogConfigXml, t.MybatisConfigXml}
}
