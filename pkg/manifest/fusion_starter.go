package manifest

type FusionStarterSpec struct {
	RawDependencies []string         `yaml:"dependencies"`
	Dependencies    []DependencyInfo `yaml:"-"`
}
