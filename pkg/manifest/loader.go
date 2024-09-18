package manifest

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
)

func LoadManifest(filePath string) (*Manifest, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// set default value
	manifest := Manifest{
		Dir: filepath.Dir(filePath),
		Main: MainSystem{
			Version: defaultMainVersion,
			JvmSize: defaultJvmSize,
			Taint: Taint{
				LogConfigXml:     defaultLogConfigXml,
				MybatisConfigXml: defaultMybatisConfigXml,
			},
		},
	}
	err = yaml.Unmarshal(data, &manifest)
	if err != nil {
		return nil, err
	}

	manifest.Main.Dependencies = parseDependencies(manifest.Main.RawDependencies)
	manifest.Main.Parent = parseDependency(manifest.Main.RawParent)

	manifest.Main.Reconcile.M = &manifest.Main

	// 设置每个组件的引用 MainSystem 并解析 dependencies 字段
	for i := range manifest.Components {
		manifest.Components[i].M = &manifest.Main
		manifest.Components[i].Dependencies = parseDependencies(manifest.Components[i].RawDependencies)
	}

	return &manifest, nil
}

func parseDependencies(rawDeps []string) []DependencyInfo {
	var parsedDependencies []DependencyInfo
	for _, dep := range rawDeps {
		parsedDependencies = append(parsedDependencies, parseDependency(dep))
	}
	return parsedDependencies
}

func parseDependency(dep string) DependencyInfo {
	parts := strings.Split(dep, ":")
	if len(parts) == 3 {
		return DependencyInfo{
			GroupId:    parts[0],
			ArtifactId: parts[1],
			Version:    parts[2],
		}
	}
	return DependencyInfo{}
}
