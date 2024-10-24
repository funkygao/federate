package manifest

type EnvironmentSpec struct {
	Name         string `yaml:"name"`
	Branch       string `yaml:"branch"`
	MavenProfile string `yaml:"mavenProfile"`
}
