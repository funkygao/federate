package manifest

import (
	"federate/pkg/util"
)

type ReconcileSpec struct {
	Taint Taint `yaml:"taint"`

	// @Bean/@Service/@Component/@RestController/MapperFactoryBean/etc
	ExcludedBeanClasses []string `yaml:"excludeClasses"`

	RpcConsumer RpcConsumerSpec        `yaml:"rpcConsumer"`
	Resources   ResourcesReconcileSpec `yaml:"resources"`

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

type Taint struct {
	LogConfigXml     string `yaml:"logConfigXml"`
	MybatisConfigXml string `yaml:"mybatisConfigXml"`
}

func (t *Taint) ResourceFiles() []string {
	return []string{t.LogConfigXml, t.MybatisConfigXml}
}
