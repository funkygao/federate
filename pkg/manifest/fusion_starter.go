package manifest

type FusionStarterSpec struct {
	RawDependencies []string         `yaml:"dependencies"`
	Dependencies    []DependencyInfo `yaml:"-"`

	Inspect InspectSpec `yaml:"inspect"`
}

type InspectSpec struct {
	AddOn []string `yaml:"addon"`
}
