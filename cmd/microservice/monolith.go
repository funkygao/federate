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
	monolithName string
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
	// Makefile .common.mk README.md inventory.yaml .gitignore
	// fusion-projects/.foo.mk

	rootDir := monolithName
	if err := os.MkdirAll(rootDir, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	generateFile(rootDir, "Makefile", "Makefile", nil)
	generateFile(rootDir, "common.mk", ".common.mk", nil)
	generateFile(rootDir, "inventory.yaml", "inventory.yaml", nil)
	generateFile(rootDir, "gitignore", ".gitignore", nil)

	projectsDir := filepath.Join(rootDir, "fusion-projects")
	if err := os.MkdirAll(projectsDir, 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}
	generateFile(projectsDir, "pom.xml", "pom.xml", nil)
}

func generateFile(rootDir, fromTemplateFile, targetFile string, data interface{}) {
	f := filepath.Join(rootDir, targetFile)
	fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, f, data)
	color.Cyan("Generated %s", f)
}

func init() {
	monolithCmd.Flags().StringVarP(&monolithName, "name", "n", "", "Name of the monolith project")
	monolithCmd.MarkFlagRequired("name")
}
