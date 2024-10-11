package manifest

import (
	"strings"

	"federate/pkg/java"
	"federate/pkg/util"
)

type MainSystem struct {
	Name          string `yaml:"name"`
	SpringProfile string `yaml:"springProfile"`
	Version       string `yaml:"version"`

	TomcatPort int16         `yaml:"tomcatPort"`
	JvmSize    string        `yaml:"jvmSize"`
	Runtime    RuntimeSpec   `yaml:"runtime"`
	MainClass  MainClassSpec `yaml:"springBootApplication"`

	RawParent string         `yaml:"parent"`
	Parent    DependencyInfo `yaml:"-"`

	RawDependencies []string         `yaml:"dependencies"`
	Dependencies    []DependencyInfo `yaml:"-"`

	Reconcile ReconcileSpec `yaml:"reconcile"`

	// Deprecated
	Features []string `yaml:"features"`
}

type RuntimeSpec struct {
	Env                 string   `yaml:"env"`
	SingletonComponents []string `yaml:"singletonClasses"`
}

type MainClassSpec struct {
	Name          string        `yaml:"class"`
	ComponentScan ComponentScan `yaml:"componentScan"`
	Imports       []string      `yaml:"import"`
	Excludes      []string      `yaml:"exclude"`
}

func (m *MainSystem) GroupId() string {
	parts := strings.Split(m.MainClass.Name, ".")
	return strings.Join(parts[:3], ".")
}

func (m *MainSystem) FederatedRuntimePackage() string {
	return java.ClassPackageName(m.MainClass.Name) + ".runtime"
}

func (m *MainSystem) HasFeature(feature string) bool {
	for _, f := range m.Features {
		if f == feature {
			return true
		}
	}
	return false
}

type ComponentScan struct {
	BasePackages  []string `yaml:"basePackages"`
	ExcludedTypes []string `yaml:"excludedTypes"`
}

type ReconcileSpec struct {
	Taint                Taint                  `yaml:"taint"`
	SingletonBeanClasses []string               `yaml:"singletonClasses"`
	ExcludedBeanClasses  []string               `yaml:"excludeClasses"`
	RpcConsumer          RpcConsumerSpec        `yaml:"rpcConsumer"`
	Resources            ResourcesReconcileSpec `yaml:"resources"`

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

type RpcConsumerSpec struct {
	IgnoreRules []IgnoreRule `yaml:"ignoreRules"`
}

type IgnoreRule struct {
	Package string   `yaml:"package"`
	Except  []string `yaml:"except"`
}

func (s *RpcConsumerSpec) IgnoreInterface(interfaceName string) bool {
	for _, rule := range s.IgnoreRules {
		if strings.HasPrefix(interfaceName, rule.Package) {
			// 检查是否在例外列表中
			for _, except := range rule.Except {
				if interfaceName == except {
					return false
				}
			}
			return true
		}
	}
	return false
}
