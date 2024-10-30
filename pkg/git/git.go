package git

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"federate/pkg/manifest"
	"github.com/fatih/color"
)

func AddSubmodules(m *manifest.Manifest) error {
	gitmodulesUpdate := false
	for _, c := range m.Components {
		// 检查 submodule 是否已存在
		cmd := exec.Command("git", "submodule", "status", c.Name)
		log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
		err := cmd.Run()
		if err == nil {
			log.Printf("git submodule[%s] already exists, skipped", c.Name)
			continue
		}

		cmd = exec.Command("git", "submodule", "add", c.Repo, c.Name)
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
	cmd := exec.Command("git", "commit", "-am", "Update .gitmodules to maintain shallow clones")
	log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to commit .gitmodules changes: %v", err)
	}
	color.Cyan(".gitmodules updated and committed")
	return nil
}
