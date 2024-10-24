package manifest

import (
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/java"
)

type Manifest struct {
	Version string `yaml:"version"`

	Main       MainSystem        `yaml:"federated"`
	Starter    FusionStarterSpec `yaml:"fusion-starter"`
	Components []ComponentInfo   `yaml:"components"`

	// Dir of the manifest file
	Dir   string            `yaml:"-"`
	State IntermediateState `yaml:"-"`
}

func (m *Manifest) StarterBaseDir() string {
	return federated.StarterBaseDir(m.Main.Name)
}

func (m *Manifest) TargetBaseDir() string {
	return m.Main.Name + "-app"
}

func (m *Manifest) ParseMainClass() (string, string) {
	parts := strings.Split(m.Main.MainClass.Name, ".")
	packageName := strings.Join(parts[:len(parts)-1], ".")
	className := parts[len(parts)-1]
	return packageName, className
}

func (m *Manifest) ComponentModules() []java.DependencyInfo {
	var dependencies []java.DependencyInfo
	for _, component := range m.Components {
		dependencies = append(dependencies, component.Modules...)
	}
	return dependencies
}

func (m *Manifest) RpmByEnv(env string) *RpmSpec {
	for _, d := range m.Main.Rpms {
		if d.Env == env {
			return &d
		}
	}
	return &RpmSpec{}
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

	for _, pattern := range component.Resources.FederatedIgnoredFiles {
		matched, err := filepath.Match(pattern, info.Name())
		if err != nil {
			log.Printf("Error on filepath.Match(pattern=%s, file=%s): %v", pattern, info.Name(), err)
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
	return filepath.Join(federated.GeneratedDir, m.Main.Name+"-app")
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
