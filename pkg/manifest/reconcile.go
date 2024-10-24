package manifest

import (
	"federate/pkg/util"
)

type ReconcileSpec struct {
	ExcludedBeanClasses []string `yaml:"excludeClasses"`

	Taint     TaintSpec              `yaml:"taint"`
	Resources ResourcesReconcileSpec `yaml:"resources"`

	Rpc RpcSpec `yaml:"rpc"`

	M *MainSystem
}

type ResourcesReconcileSpec struct {
	// xml 里定义的 bean
	SingletonBeanClasses []string               `yaml:"singletonClasses"`
	FlatCopy             []string               `yaml:"flatCopy"`
	PropertySettlement   map[string]interface{} `yaml:"propertySettlement"`
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
