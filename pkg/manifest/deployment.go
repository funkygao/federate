package manifest

type DeploymentSpec struct {
	Env        string `yaml:"env"`
	TomcatPort int16  `yaml:"tomcatPort"`
	JvmSize    string `yaml:"jvmSize"`
	JavaOpts   string `yaml:"JAVA_OPTS"`
}
