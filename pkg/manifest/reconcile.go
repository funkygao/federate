package manifest

import (
	"federate/pkg/util"
)

type ReconcileSpec struct {
	Taint Taint `yaml:"taint"`

	// @Bean/@Service/@Component/@RestController/MapperFactoryBean/etc
	ExcludedBeanClasses []string `yaml:"excludeClasses"`

	RpcConsumer        RpcConsumerSpec        `yaml:"rpcConsumer"`
	Resources          ResourcesReconcileSpec `yaml:"resources"`
	PropertySettlement map[string]string      `yaml:"propertySettlement"`

	M *MainSystem
}

type ResourcesReconcileSpec struct {
	// xml 里定义的 bean
	SingletonBeanClasses []string `yaml:"singletonClasses"`
	FlatCopy             []string `yaml:"flatCopy"`
}

func (s *ReconcileSpec) ExcludeBeanClass(class string) bool {
	return util.Contains(class, s.ExcludedBeanClasses)
}

func (s *ReconcileSpec) SingletonBeanClass(class string) bool {
	return util.Contains(class, s.Resources.SingletonBeanClasses)
}

type Taint struct {
	LogConfigXml     string `yaml:"logConfigXml"`
	MybatisConfigXml string `yaml:"mybatisConfigXml"`
}

func (t *Taint) ResourceFiles() []string {
	return []string{t.LogConfigXml, t.MybatisConfigXml}
}
