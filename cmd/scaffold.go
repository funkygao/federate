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
	generateMakefile(m)
	color.Green("🍺 Starter scaffold generated for federated system: %s", m.Main.Name)
}

func generatePomFile(m *manifest.Manifest) {
	data := struct {
		Name                  string
		ComponentDependencies []manifest.DependencyInfo
	}{
		Name:                  m.Main.Name,
		ComponentDependencies: m.ComponentDependencies(),
	}
	fn := filepath.Join(m.Dir, "pom.xml")
	fs.GenerateFileFromTmpl("templates/starter.pom.xml", fn, data)
	color.Cyan("Generated %s", fn)
}

func generateMakefile(m *manifest.Manifest) {
	data := struct {
		AppName string
	}{
		AppName: m.Main.Name,
	}
	fn := filepath.Join(m.Dir, "Makefile")
	fs.GenerateFileFromTmpl("templates/starter.Makefile", fn, data)
	color.Cyan("Generated %s", fn)
}

func init() {
	addRequiredInputFlag(scaffoldCmd)
}
