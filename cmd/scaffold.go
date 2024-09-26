package cmd

import (
	"log"
	"path/filepath"

	"federate/internal/fs"
	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var projectName string

var scaffoldCmd = &cobra.Command{
	Use:   "create",
	Short: "Scaffold a new federated target system",
	Long: `The create command scaffolds a new federated target system.

Example usage:
  federate microservice create -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m, err := manifest.LoadManifest(manifestFile)
		if err != nil {
			log.Fatalf("Error loading manifest: %v", err)
		}

		scaffoldProject(m)
	},
}

func scaffoldProject(m *manifest.Manifest) {
	generatePomFile(m)
}

func generatePomFile(m *manifest.Manifest) {
	pomData := struct {
		Name                  string
		ComponentDependencies []manifest.DependencyInfo
	}{
		Name:                  m.Main.Name,
		ComponentDependencies: m.ComponentDependencies(),
	}
	fn := filepath.Join(m.Dir, "pom.xml")
	fs.GenerateFileFromTmpl("templates/starter.pom.xml", fn, pomData)
	color.Cyan("Generated %s", fn)
}

func init() {
	addRequiredInputFlag(scaffoldCmd)
}
