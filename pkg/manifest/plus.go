package manifest

type PlusSpec struct {
	BasePackage     string `yaml:"basePackage"`
	EntryPointClass string `yaml:"entryPoint"`
	SpringXml       string `yaml:"springXml"`
}
