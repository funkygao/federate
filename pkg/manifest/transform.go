package manifest

type TransformSpec struct {
	Beans     map[string]string `yaml:"beans"`
	Autowired AutowiredSpec     `yaml:"autowired"`
}

type AutowiredSpec struct {
}
