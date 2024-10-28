package microservice

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"federate/cmd/image"
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
	generateJdosDockerfile(m)

	// create starter dir
	if err := os.MkdirAll(federated.StarterBaseDir(m.Main.Name), 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	color.Green("🍺 Fusion project[%s] scaffolded. You MUST add user:JDOSBOOT for this git repo!", m.Main.Name)
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, targetFile, data)
	if overwrite {
		color.Yellow("Overwrite %s", targetFile)
	} else {
		color.Cyan("Generated %s", targetFile)
	}
}

func generateJdosDockerfile(m *manifest.Manifest) {
	fn := "Dockerfile"
	overwrite := image.GenerateJdosDockerfile(m, fn)
	if !overwrite {
		color.Cyan("Generated %s", fn)
	} else {
		color.Yellow("Overwrite %s", fn)
	}

	// Git add Dockerfile
	addCmd := exec.Command("git", "add", fn)
	log.Printf("Executing: %s", strings.Join(addCmd.Args, " "))
	err := addCmd.Run()
	if err != nil {
		log.Printf("Warning: failed to git add %s: %v", fn, err)
		return
	}

	// Git commit Dockerfile
	commitCmd := exec.Command("git", "commit", "-m", fmt.Sprintf("Added JDOS %s", fn))
	log.Printf("Executing: %s", strings.Join(commitCmd.Args, " "))
	err = commitCmd.Run()
	if err != nil {
		log.Printf("Warning: failed to commit %s: %v", fn, err)
		return
	}

	color.Cyan("%s added and committed", fn)
}

func addGitSubmodules(m *manifest.Manifest) error {
	gitmodulesUpdate := false
	for _, c := range m.Components {
		// 检查 submodule 是否已存在
		checkCmd := exec.Command("git", "submodule", "status", c.Name)
		err := checkCmd.Run()
		if err == nil {
			// Submodule 已存在，跳过
			continue
		}

		cmd := exec.Command("git", "submodule", "add", c.Repo, c.Name)
		log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		err = cmd.Run()
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
