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
	"federate/pkg/util"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Generate a Plus Project with standard structure and boilerplate code",
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

	log.Printf("Scaffolding %s project structure ...", m.Main.Name)
	generatePlusProjectFiles(m)

	log.Println("Instrumenting submodule pom.xml ...")
	merge.EchoBeer = false
	merge.InstrumentPomForFederatePackaging(m)
	color.Green("🍺 Congrat, %s scaffolded! Next, `make install-kernel` and start programming!", m.Main.Name)
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
	if util.FileExists(targetFile) {
		log.Printf("%s exists, skipped to avoid being overwritten", targetFile)
		return
	}

	fs.GenerateFileFromTmpl("templates/plus/"+fromTemplateFile, targetFile, data)
	log.Printf("Generated %s", targetFile)
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
