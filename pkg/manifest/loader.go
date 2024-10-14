package manifest

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

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
		Version: "1.0",
		Dir:     filepath.Dir(filePath),
		Main: MainSystem{
			Version: defaultMainVersion,
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

	manifest.Main.Dependency.Includes = parseDependencies(manifest.Main.Dependency.RawInclude)
	manifest.Main.Dependency.Excludes = parseDependencies(manifest.Main.Dependency.RawExclude)
	manifest.Starter.Dependencies = parseDependencies(manifest.Starter.RawDependencies)
	manifest.Main.Parent = parseDependency(manifest.Main.RawParent)

	manifest.Main.Reconcile.M = &manifest.Main

	// 设置每个组件的引用 MainSystem 并解析 dependencies 字段
	for i := range manifest.Components {
		manifest.Components[i].M = &manifest.Main
		manifest.Components[i].Dependencies = parseDependencies(manifest.Components[i].RawDependencies)
	}

	return &manifest
}
