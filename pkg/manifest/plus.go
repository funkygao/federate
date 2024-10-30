package manifest

type PlusSpec struct {
	BasePackage string           `yaml:"basePackage"`
	Resource    PlusResourceSpec `yaml:"resources"`
}

type PlusResourceSpec struct {
}
