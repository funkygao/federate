package manifest

import (
	"path/filepath"

	"federate/pkg/federated"
)

type ComponentInfo struct {
	Name                  string           `yaml:"name"`
	SpringProfile         string           `yaml:"springProfile"`
	RawDependencies       []string         `yaml:"modules"`
	Dependencies          []DependencyInfo `yaml:"-"`
	ResourceBaseDirs      []string         `yaml:"resourceBasedirs"`
	ImportSpringXML       []string         `yaml:"importSpringXmls"`
	DubboConsumerXmls     []string         `yaml:"dubboConsumerXmls"`
	JsfConsumerXmls       []string         `yaml:"jsfConsumerXmls"`
	PropertySources       []string         `yaml:"propertySources"`
	FederatedIgnoredFiles []string         `yaml:"federatedIgnore"`

	// BaseDir is used for unit test: change source dir
	BaseDir string

	M *MainSystem
}

// 该组件的源代码根目录
func (c *ComponentInfo) RootDir() string {
	if c.BaseDir != "" {
		return filepath.Join(c.BaseDir, c.Name)
	}
	return filepath.Join(c.Name)
}

// 该组件的基于 baseDir 的源代码目录
func (c *ComponentInfo) SrcDir(baseDir string) string {
	if c.BaseDir != "" {
		return filepath.Join(c.BaseDir, c.Name, baseDir)
	}
	return filepath.Join(c.Name, baseDir)
}

// 合并生成的资源文件目录，e.g, generated/{project}/src/main/resources/federated/{component}
func (c *ComponentInfo) TargetResourceDir() string {
	if c.BaseDir != "" {
		return filepath.Join(c.BaseDir, federated.GeneratedResourceBaseDir(c.M.Name), c.Name)
	}
	return filepath.Join(federated.GeneratedResourceBaseDir(c.M.Name), c.Name)
}

func (c *ComponentInfo) JSFEnabled() bool {
	return len(c.JsfConsumerXmls) > 0
}

func (c *ComponentInfo) DubboEnabled() bool {
	return len(c.DubboConsumerXmls) > 0
}
