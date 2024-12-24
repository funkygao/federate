package manifest

import (
	"os"
	"strings"
)

type PlusSpec struct {
	BasePackage     string        `yaml:"basePackage"`
	EntryPointClass string        `yaml:"entryPoint"`
	SpringXml       string        `yaml:"springXml"`
	Obfuscate       ObfuscateSpec `yaml:"obfuscate"`
}

type ObfuscateSpec struct {
	Jars []string `yaml:"jars"`
}

func (p *PlusSpec) ObfuscateJar(info os.FileInfo) bool {
	if !strings.HasSuffix(info.Name(), ".jar") {
		return false
	}

	for _, jar := range p.Obfuscate.Jars {
		if strings.HasPrefix(info.Name(), jar+"-") {
			return true
		}
	}

	return false
}
