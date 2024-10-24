package manifest

import (
	"os"
	"path/filepath"

	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/util"
)

type ComponentInfo struct {
	Name          string `yaml:"name"`
	SpringProfile string `yaml:"springProfile"`

	RawDependencies []string              `yaml:"modules"`
	Dependencies    []java.DependencyInfo `yaml:"-"`

	Resources ComponentResourceSpec `yaml:"resources"`

	// BaseDir is used for unit test: change source dir
	BaseDir string `yaml:"-"`

	M *MainSystem
}

type ComponentResourceSpec struct {
	BaseDirs              []string `yaml:"baseDir"`
	ImportSpringXML       []string `yaml:"import"`
	DubboConsumerXmls     []string `yaml:"dubboConsumerXml"`
	JsfConsumerXmls       []string `yaml:"jsfConsumerXml"`
	PropertySources       []string `yaml:"propertySource"`
	FederatedIgnoredFiles []string `yaml:"ignore"`
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

// 返回的目录名都是相对路径，不包含 RootDir 信息
func (c *ComponentInfo) ChildDirs() (childDirs []string) {
	entries, _ := os.ReadDir(c.RootDir())
	for _, entry := range entries {
		if entry.IsDir() {
			childDirs = append(childDirs, entry.Name())
		}
	}

	return childDirs
}

// 返回的路径是已经包含了 RootDir 信息
func (c *ComponentInfo) MavenModules() (dirs []string) {
	for _, d := range c.ChildDirs() {
		if util.FileExists(filepath.Join(c.RootDir(), d, "pom.xml")) {
			dirs = append(dirs, filepath.Join(c.RootDir(), d))
		}
	}
	return
}

// 合并生成的资源文件目录，e.g, generated/{project}/src/main/resources/federated/{component}
func (c *ComponentInfo) TargetResourceDir() string {
	if c.BaseDir != "" {
		return filepath.Join(c.BaseDir, federated.GeneratedResourceBaseDir(c.M.Name), c.Name)
	}
	return filepath.Join(federated.GeneratedResourceBaseDir(c.M.Name), c.Name)
}

func (c *ComponentInfo) JSFEnabled() bool {
	return len(c.Resources.JsfConsumerXmls) > 0
}

func (c *ComponentInfo) DubboEnabled() bool {
	return len(c.Resources.DubboConsumerXmls) > 0
}
