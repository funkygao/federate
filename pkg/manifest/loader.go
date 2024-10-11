package manifest

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

var filePath string

// RequiredManifestFileFlag adds the --input flag to the given command and marks it as required
func RequiredManifestFileFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&filePath, "input", "i", "", "Path to the manifest file")
	cmd.MarkFlagRequired("input")
}

func LoadManifest() *Manifest {
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}

	// set default value
	manifest := Manifest{
		Dir: filepath.Dir(filePath),
		Main: MainSystem{
			Version: defaultMainVersion,
			JvmSize: defaultJvmSize,
			Reconcile: ReconcileSpec{
				Taint: Taint{
					LogConfigXml:     defaultLogConfigXml,
					MybatisConfigXml: defaultMybatisConfigXml,
				},
			},
		},
	}
	err = yaml.Unmarshal(data, &manifest)
	if err != nil {
		log.Fatalf("Error loading manifest: %v", err)
	}

	manifest.Main.Dependencies = parseDependencies(manifest.Main.RawDependencies)
	manifest.Main.Starter.Dependencies = parseDependencies(manifest.Main.Starter.RawDependencies)
	manifest.Main.Parent = parseDependency(manifest.Main.RawParent)

	manifest.Main.Reconcile.M = &manifest.Main

	// 设置每个组件的引用 MainSystem 并解析 dependencies 字段
	for i := range manifest.Components {
		manifest.Components[i].M = &manifest.Main
		manifest.Components[i].Dependencies = parseDependencies(manifest.Components[i].RawDependencies)
	}

	return &manifest
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
			Scope:      "compile",
		}
	}
	if len(parts) == 4 {
		return DependencyInfo{
			GroupId:    parts[0],
			ArtifactId: parts[1],
			Version:    parts[2],
			Scope:      parts[3],
		}
	}
	return DependencyInfo{}
}
