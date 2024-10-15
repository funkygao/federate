package microservice

import (
	"log"
	"os"
	"path/filepath"

	"federate/internal/fs"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	monolithName       string
	fusionProjectsName string
)

var monolithCmd = &cobra.Command{
	Use:   "scaffold-monolith",
	Short: "Scaffold a logical monolith from multiple existing code repositories",
	Long: `The monolith command scaffolds a logical monolithic code repository by integrating 
multiple existing code repositories using git submodules.

This approach allows you to:
1. Manage multiple microservices as a single codebase
2. Offload the decisions of how to deploy to federate phase
3. Preserve the existing development workflow without disruption`,
	Run: func(cmd *cobra.Command, args []string) {
		createMonolith()
	},
}

func createMonolith() {
	rootDir := monolithName
	if err := os.MkdirAll(rootDir, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	data := struct {
		FusionProjectsName string
	}{
		FusionProjectsName: fusionProjectsName,
	}
	generateFile(rootDir, "Makefile", "Makefile", data)
	generateFile(rootDir, "common.mk", ".common.mk", data)
	generateFile(rootDir, "inventory.yaml", "inventory.yaml", data)
	generateFile(rootDir, "gitignore", ".gitignore", data)

	projectsDir := filepath.Join(rootDir, "fusion-projects")
	if err := os.MkdirAll(projectsDir, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}
	generateFile(projectsDir, "pom.xml", "pom.xml", data)
	generateFile(projectsDir, "demo.mk", ".demo.mk", data)

	demoFusionStarter := filepath.Join(projectsDir, "demo")
	if err := os.MkdirAll(demoFusionStarter, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}
	generateFile(demoFusionStarter, "manifest.yaml", "manifest.yaml", data)

	color.Green("🍺 Monolith project[%s] scaffolded with all fusion-starter projects reside in %s/", monolithName, fusionProjectsName)
}

func generateFile(rootDir, fromTemplateFile, targetFile string, data interface{}) {
	f := filepath.Join(rootDir, targetFile)
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, f, data)
	if overwrite {
		color.Yellow("Overwrite %s", f)
	} else {
		color.Cyan("Generated %s", f)
	}
}

func init() {
	monolithCmd.Flags().StringVarP(&monolithName, "name", "n", "", "Name of the monolith project")
	monolithCmd.MarkFlagRequired("name")
	monolithCmd.Flags().StringVarP(&fusionProjectsName, "fusion-projects", "p", "fusion-projects", "In which directory will all fusion-starter projects reside")
}
