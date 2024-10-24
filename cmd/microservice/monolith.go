package microservice

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"federate/internal/fs"
	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var monolithCmd = &cobra.Command{
	Use:   "scaffold",
	Short: "Scaffold a logical monolith from multiple existing code repositories",
	Long: `The monolith command scaffolds a logical monolithic code repository by integrating 
multiple existing code repositories using git submodules.`,
	Run: func(cmd *cobra.Command, args []string) {
		scaffoldMonolith()
	},
}

func scaffoldMonolith() {
	log.Printf("Parsing %s to configure git submodule", manifest.File())

	m := manifest.Load()

	// 添加 git submodules
	if err := addGitSubmodules(m); err != nil {
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
	}{
		FusionProjectName: m.Main.Name,
		FusionStarter:     federated.StarterBaseDir(m.Main.Name),
		Parent:            m.Main.Parent,
		GroupId:           m.Main.GroupId,
	}
	generateFile("Makefile", "Makefile", data)
	generateFile("pom.xml", "pom.xml", data)
	generateFile("gitignore", ".gitignore", data)

	// create starter dir
	if err := os.MkdirAll(federated.StarterBaseDir(m.Main.Name), 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	color.Green("🍺 Fusion project[%s] scaffolded.", m.Main.Name)
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, targetFile, data)
	if overwrite {
		color.Yellow("Overwrite %s", targetFile)
	} else {
		color.Cyan("Generated %s", targetFile)
	}
}

func addGitSubmodules(m *manifest.Manifest) error {
	gitmodulesUpdate := false
	for _, c := range m.Components {
		cmd := exec.Command("git", "submodule", "add", c.Repo, c.Name)
		log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		err := cmd.Run()
		if err == nil {
			color.Cyan("Added git submodule: %s", c.Name)

			gitmodulesUpdate = true
		}
	}

	if !gitmodulesUpdate {
		return nil
	}

	// 提交 .gitmodules 更改
	commitCmd := exec.Command("git", "commit", "-am", "Update .gitmodules to maintain shallow clones")
	log.Printf("Executing: %s", strings.Join(commitCmd.Args, " "))
	err := commitCmd.Run()
	if err != nil {
		return fmt.Errorf("failed to commit .gitmodules changes: %v", err)
	}
	color.Cyan(".gitmodules updated and committed")
	return nil
}

func init() {
	manifest.RequiredManifestFileFlag(monolithCmd)
}
