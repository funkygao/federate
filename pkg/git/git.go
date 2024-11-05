package git

import (
	"fmt"
	"log"
	"os/exec"
	"strings"

	"federate/pkg/manifest"
)

func AddSubmodules(m *manifest.Manifest) error {
	gitmodulesUpdate := false
	for _, c := range m.Components {
		// 检查 submodule 是否已存在
		cmd := exec.Command("git", "submodule", "status", c.Name)
		err := cmd.Run()
		if err == nil {
			log.Printf("git submodule[%s] already exists, skipped", c.Name)
			continue
		}

		cmd = exec.Command("git", "submodule", "add", c.Repo, c.Name)
		log.Printf("Executing: %s", strings.Join(cmd.Args, " "))
		err = cmd.Run()
		if err == nil {
			gitmodulesUpdate = true
		}
	}

	// 让每个 submodule 的最新commit ID/分支，各自独立，在 git merge 时不进行合并
	cmd := exec.Command("git", "config", "merge.keep-local.driver", "true")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to enable merge.keep-local.driver: %v", err)
	}

	if gitmodulesUpdate {
		// 提交 .gitmodules 更改
		cmd := exec.Command("git", "commit", "-am", "Update .gitmodules")
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to commit .gitmodules changes: %v", err)
		}
	}

	return nil
}
