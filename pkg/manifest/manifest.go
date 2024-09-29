package manifest

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/util"
)

type Manifest struct {
	Main       MainSystem      `yaml:"target"`
	Components []ComponentInfo `yaml:"components"`

	// Dir of the manifest file
	Dir   string            `yaml:"-"`
	State IntermediateState `yaml:"-"`
}

func (m *Manifest) ParseMainClass() (string, string) {
	parts := strings.Split(m.Main.MainClass.Name, ".")
	packageName := strings.Join(parts[:len(parts)-1], ".")
	className := parts[len(parts)-1]
	return packageName, className
}

func (m *Manifest) ComponentDependencies() []DependencyInfo {
	var dependencies []DependencyInfo
	for _, component := range m.Components {
		dependencies = append(dependencies, component.Dependencies...)
	}
	return dependencies
}

func (m *Manifest) FirstComponent() *ComponentInfo {
	return &m.Components[0]
}

func (m *Manifest) ComponentByName(componentName string) *ComponentInfo {
	for _, c := range m.Components {
		if c.Name == componentName {
			return &c
		}
	}
	return nil
}

// Ignore component source file
func (m *Manifest) IgnoreResourceSrcFile(info os.FileInfo, component ComponentInfo) bool {
	if info.IsDir() {
		return false
	}

	if m.State.isMergeSource(info.Name(), component) {
		return true
	}

	for _, pattern := range m.Main.Reconcile.IgnoredFiles {
		matched, err := filepath.Match(pattern, info.Name())
		if err != nil {
			return false
		}
		if matched {
			return true
		}
	}
	return false
}

func (m *Manifest) HasFeature(feature string) bool {
	return m.Main.HasFeature(feature)
}

func (m *Manifest) JSFEnabled() bool {
	for _, component := range m.Components {
		if component.JSFEnabled() {
			return true
		}
	}
	return false
}

func (m *Manifest) DubboEnabled() bool {
	for _, component := range m.Components {
		if component.DubboEnabled() {
			return true
		}
	}
	return false
}

func (m *Manifest) CreateTargetSystemDir() (string, error) {
	dir := m.TargetRootDir()
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return "", err
	}
	return dir, nil
}

func (m *Manifest) TargetRootDir() string {
	return filepath.Join(federated.GeneratedDir, m.Main.Name)
}

func (m *Manifest) TargetResourceDir() string {
	return filepath.Join(m.TargetRootDir(), "src", "main", "resources")
}

type IntermediateState struct {
	mergeSources map[string]struct{}
}

func (s *IntermediateState) AddMergeSource(file string, component ComponentInfo) error {
	if len(s.mergeSources) == 0 {
		s.mergeSources = make(map[string]struct{})
	}
	if s.isMergeSource(file, component) {
		return fmt.Errorf("Resource[%s] already added as merge source", file)
	}
	s.addMergeSource(file, component)
	return nil
}

func (s *IntermediateState) addMergeSource(file string, component ComponentInfo) {
	key := component.Name + ":" + path.Base(file)
	s.mergeSources[key] = struct{}{}
}

func (s *IntermediateState) isMergeSource(file string, component ComponentInfo) bool {
	basename := path.Base(file)
	for key, _ := range s.mergeSources {
		if strings.HasPrefix(key, component.Name+":") && strings.HasSuffix(key, basename) {
			return true
		}
	}
	return false
}

type MainSystem struct {
	Name          string `yaml:"name"`
	SpringProfile string `yaml:"springProfile"`
	Version       string `yaml:"version"`

	TomcatPort int16       `yaml:"tomcatPort"`
	JvmSize    string      `yaml:"jvmSize"`
	Runtime    RuntimeSpec `yaml:"runtime"`
	MainClass  MainClass   `yaml:"mainClass"`

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

type MainClass struct {
	Name          string        `yaml:"name"`
	ComponentScan ComponentScan `yaml:"componentScan"`
	Imports       []string      `yaml:"import"`
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
	Logger               string   `yaml:"logger"`
	Taint                Taint    `yaml:"taint"`
	SingletonBeanClasses []string `yaml:"singletonClasses"`
	ExcludedBeanClasses  []string `yaml:"excludeClasses"`
	MergeResourceFiles   []string `yaml:"mergeResources"`
	IgnoredFiles         []string `yaml:"ignoreResources"`

	M *MainSystem
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

type ComponentInfo struct {
	Name              string           `yaml:"name"`
	SpringProfile     string           `yaml:"springProfile"` // TODO auto detect from spring.profiles.active
	RawDependencies   []string         `yaml:"dependencies"`
	Dependencies      []DependencyInfo `yaml:"-"`
	ResourceBaseDirs  []string         `yaml:"resourceBasedirs"`
	ImportSpringXML   []string         `yaml:"importSpringXmls"`
	DubboConsumerXmls []string         `yaml:"dubboConsumerXmls"`
	JsfConsumerXmls   []string         `yaml:"jsfConsumerXmls"`
	PropertySources   []string         `yaml:"propertySources"`
	DataSource        DataSource       `yaml:"dataSource"`

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

type DependencyInfo struct {
	GroupId    string
	ArtifactId string
	Version    string
}

type DataSource struct {
}
