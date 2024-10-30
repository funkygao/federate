package plus

import (
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/cmd/merge"
	"federate/internal/fs"
	"federate/pkg/git"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Generate a new plus project with standard structure",
	Long:  `Scaffold a new plus project, laying the foundation for platform extensions with an optimized directory structure and essential boilerplate code.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doCreate(m)
	},
}

func doCreate(m *manifest.Manifest) {
	// 添加 git submodules
	if err := git.AddSubmodules(m); err != nil {
		log.Fatalf("Error adding git submodules: %v", err)
	}

	generatePlusProjectFiles(m)

	log.Println("Instrument submodule pom.xml ...")
	merge.InstrumentPomForFederatePackaging(m)
}

func generatePlusProjectFiles(m *manifest.Manifest) {
	data := struct {
		ArtifactId            string
		ComponentDependencies []java.DependencyInfo
	}{
		ArtifactId:            m.Main.Name,
		ComponentDependencies: m.ComponentModules(),
	}
	generateFile("pom.xml", "pom.xml", data)
	generateFile("Makefile", "Makefile", data)

	paths := [][]string{
		{"src", "main", "java", filepath.FromSlash(strings.ReplaceAll(m.Main.Plus.BasePackage, ".", "/")), "plus", "controller"},
		{"src", "main", "java", filepath.FromSlash(strings.ReplaceAll(m.Main.Plus.BasePackage, ".", "/")), "plus", "policy"},
		{"src", "main", "java", filepath.FromSlash(strings.ReplaceAll(m.Main.Plus.BasePackage, ".", "/")), "plus", "pattern"},
		{"src", "main", "resources"},
	}
	for _, p := range paths {
		mkdir(filepath.Join(p...))
	}
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	overwrite := fs.GenerateFileFromTmpl("templates/plus/"+fromTemplateFile, targetFile, data)
	if overwrite {
		color.Yellow("Overwrite %s", targetFile)
	} else {
		color.Cyan("Generated %s", targetFile)
	}
}

func mkdir(path string) {
	log.Printf("Generating %s", path)
	if err := os.MkdirAll(path, 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}
}

func init() {
	manifest.RequiredManifestFileFlag(createCmd)
}
