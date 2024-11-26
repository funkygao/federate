package microservice

import (
	"log"
	"os"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/git"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var scaffoldCmd = &cobra.Command{
	Use:   "scaffold",
	Short: "Scaffold a fusion project from multiple existing code repositories",
	Long: `The scaffold command scaffolds a fusion project code repository by integrating 
multiple existing code repositories using git submodules.`,
	Run: func(cmd *cobra.Command, args []string) {
		scaffoldMonolith()
	},
}

func scaffoldMonolith() {
	log.Printf("Parsing %s to configure git submodule", manifest.File())

	m := manifest.Load()

	// 添加 git submodules
	if err := git.AddSubmodules(m); err != nil {
		log.Fatalf("Error adding git submodules: %v", err)
	}

	generateMonolithFiles(m)
}

func generateMonolithFiles(m *manifest.Manifest) {
	data := struct {
		FusionProjectName string
		FusionStarter     string
		Parent            java.DependencyInfo
		GroupId           string
		Components        []string
	}{
		FusionProjectName: m.Main.Name,
		FusionStarter:     federated.StarterBaseDir(m.Main.Name),
		Parent:            m.Main.Parent,
		GroupId:           m.Main.GroupId,
		Components:        m.ComponentNames(),
	}
	generateFile("Makefile", "Makefile", data)
	generateFile("pom.xml", "pom.xml", data)
	generateFile("gitignore", ".gitignore", data)
	// Add .gitattributes to skip merging specific files
	generateFile("gitattributes", ".gitattributes", data)

	// create starter dir
	if err := os.MkdirAll(federated.StarterBaseDir(m.Main.Name), 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	color.Green("🍺 Fusion project[%s] scaffolded.", m.Main.Name)
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, targetFile, data)
}

func init() {
	manifest.RequiredManifestFileFlag(scaffoldCmd)
}
