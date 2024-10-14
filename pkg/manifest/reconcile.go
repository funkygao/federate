package manifest

import (
	"federate/pkg/util"
)

type ReconcileSpec struct {
	Taint                Taint                  `yaml:"taint"`
	SingletonBeanClasses []string               `yaml:"singletonClasses"`
	ExcludedBeanClasses  []string               `yaml:"excludeClasses"`
	RpcConsumer          RpcConsumerSpec        `yaml:"rpcConsumer"`
	Resources            ResourcesReconcileSpec `yaml:"resources"`
	PropertyOverrides    ResourcesReconcileSpec `yaml:"federatedPropertyOverrides"`

	M *MainSystem
}

type ResourcesReconcileSpec struct {
	FlatCopy []string `yaml:"flatCopy"`
}

func (s *ReconcileSpec) ExcludeBeanClass(class string) bool {
	return util.Contains(class, s.ExcludedBeanClasses)
}

func (s *ReconcileSpec) SingletonBeanClass(class string) bool {
	return util.Contains(class, s.SingletonBeanClasses)
}

type Taint struct {
	LogConfigXml     string `yaml:"logConfigXml"`
	MybatisConfigXml string `yaml:"mybatisConfigXml"`
}

func (t *Taint) ResourceFiles() []string {
	return []string{t.LogConfigXml, t.MybatisConfigXml}
}
