package manifest

import (
	"federate/pkg/java"
)

type MainSystem struct {
	Name          string `yaml:"name"`
	GroupId       string `yaml:"groupId"`
	SpringProfile string `yaml:"springProfile"`
	Version       string `yaml:"version"`

	Runtime   RuntimeSpec   `yaml:"runtime"`
	MainClass MainClassSpec `yaml:"springBootApplication"`

	RawParent string              `yaml:"parent"`
	Parent    java.DependencyInfo `yaml:"-"`

	Dependency MainDependencySpec `yaml:"dependency"`

	Reconcile ReconcileSpec `yaml:"reconcile"`

	Rpms []RpmSpec `yaml:"rpm"`
	Plus PlusSpec  `yaml:"plus"`

	// Deprecated
	Features []string `yaml:"features"`
}

type MainDependencySpec struct {
	RawInclude []string `yaml:"include"`
	RawExclude []string `yaml:"exclude"`

	Includes []java.DependencyInfo `yaml:"-"`
	Excludes []java.DependencyInfo `yaml:"-"`
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

func (m *MainSystem) FederatedRuntimePackage() string {
	return java.ClassPackageName(m.MainClass.Name) + ".runtime"
}

func (m *MainSystem) PlusBasePackage() string {
	return m.Plus.BasePackage + ".plus"
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
