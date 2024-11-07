package manifest

import (
	"federate/pkg/java"
)

type FusionStarterSpec struct {
	ExitOnStartupFailure bool `yaml:"exitOnStartupFailure"`

	RawDependencies []string              `yaml:"dependencies"`
	Dependencies    []java.DependencyInfo `yaml:"-"`

	Inspect InspectSpec `yaml:"inspect"`

	ResourceLoader    ResourceLoaderSpec    `yaml:"resourceLoader"`
	BeanNameGenerator BeanNameGeneratorSpec `yaml:"beanNameGenerator"`
}

type InspectSpec struct {
	AddOn []string `yaml:"addon"`
}

type ResourceLoaderSpec struct {
}

type BeanNameGeneratorSpec struct {
	ExcludedBeanPatterns []string `yaml:"excludedBeanPatterns"`
}
