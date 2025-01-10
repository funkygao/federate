package git

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func GitHistory(filename string) (int, error) {
	repoPath := filepath.Dir(filename)
	for {
		if _, err := filepath.Abs(filepath.Join(repoPath, ".git")); err == nil {
			break
		}

		parent := filepath.Dir(repoPath)
		if parent == repoPath {
			return 0, fmt.Errorf("not a git repository")
		}
		repoPath = parent
	}

	// 获取文件相对于仓库根目录的路径
	relPath, err := filepath.Rel(repoPath, filename)
	if err != nil {
		return 0, err
	}

	// 执行 git 命令获取提交历史次数
	cmd := exec.Command("git", "-C", repoPath, "rev-list", "--count", "HEAD", "--", relPath)
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	commits := strings.TrimSpace(string(output))
	n, err := strconv.Atoi(commits)
	if err != nil {
		return 0, err
	}

	return n, nil
}
