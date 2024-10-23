package microservice

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/inventory"
	"federate/pkg/step"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	inventoryFile string
)

var monolithCmd = &cobra.Command{
	Use:   "scaffold-monolith",
	Short: "Scaffold a logical monolith from multiple existing code repositories",
	Long: `The monolith command scaffolds a logical monolithic code repository by integrating 
multiple existing code repositories using git submodules.`,
	Run: func(cmd *cobra.Command, args []string) {
		scaffoldMonolith()
	},
}

func scaffoldMonolith() {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Error getting current working directory: %v", err)
	}
	monolithName := filepath.Base(cwd)
	if !step.ConfirmAction(fmt.Sprintf("You are about to create a fusion project named '%s'. Is this correct?", monolithName)) {
		fmt.Println("Operation cancelled.")
		return
	}

	log.Printf("Parsing %s to config git submodule", inventoryFile)

	// 读取和解析 inventory.yaml
	inv, err := inventory.ReadInventory(inventoryFile)
	if err != nil {
		log.Fatalf("Error reading inventory file: %v", err)
	}

	// 添加 git submodules
	err = addGitSubmodules(inv)
	if err != nil {
		log.Fatalf("Error adding git submodules: %v", err)
	}

	generateMonolithFiles(monolithName)
}

func generateMonolithFiles(monolithName string) {
	data := struct {
		Inventory         string
		FusionProjectName string
		FusionStarter     string
	}{
		Inventory:         inventoryFile,
		FusionProjectName: monolithName,
		FusionStarter:     "fusion-starter",
	}
	generateFile("Makefile", "Makefile", data)
	generateFile("common.mk", ".common.mk", data)
	generateFile("inventory.yaml", "inventory.yaml", data)
	generateFile("gitignore", ".gitignore", data)
	generateFile("pom.xml", "pom.xml", data)

	//demoFusionStarter := filepath.Join(projectsDir, "demo")
	//if err := os.MkdirAll(demoFusionStarter, 0755); err != nil {
	//	log.Fatalf("Error creating directory: %v", err)
	//}
	//generateFile(demoFusionStarter, "manifest.yaml", "manifest.yaml", data)

	color.Green("🍺 Monolith project[%s] scaffolded. Next, craft your manifest.yaml", monolithName)
}

func generateFile(fromTemplateFile, targetFile string, data interface{}) {
	overwrite := fs.GenerateFileFromTmpl("templates/fusion-project/"+fromTemplateFile, targetFile, data)
	if overwrite {
		color.Yellow("Overwrite %s", targetFile)
	} else {
		color.Cyan("Generated %s", targetFile)
	}
}

func addGitSubmodules(inv *inventory.Inventory) error {
	gitmodulesUpdate := false
	for name, repo := range inv.Repos {
		cmd := exec.Command("git", "submodule", "add", "--depth", "1", repo.Address, name)
		log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		err := cmd.Run()
		if err == nil {
			color.Cyan("Added git submodule: %s", name)

			gitmodulesUpdate = true

			// 更新 .gitmodules 文件以保持浅克隆
			updateCmd := exec.Command("git", "config", "-f", ".gitmodules", fmt.Sprintf("submodule.%s.shallow", name), "true")
			log.Printf("Executing: %s", strings.Join(updateCmd.Args, " "))
			err = updateCmd.Run()
			if err != nil {
				return fmt.Errorf("failed to update .gitmodules for %s: %v", name, err)
			}
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
	monolithCmd.Flags().StringVarP(&inventoryFile, "inventory", "i", "inventory.yaml", "Path to the inventory file specifying which source code repositories to consolidate")
	if inventoryFile == "" {
		log.Fatal("required flag --inventory not set")
	}
}
