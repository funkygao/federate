package onpremise

import (
	"fmt"
	"os"
	"os/user"
	"strings"
)

// containsWarningsOrErrors 检查 Ansible 输出内容中是否包含警告或错误
func containsWarningsOrErrors(output string) bool {
	return strings.Contains(output, "[WARNING]") || strings.Contains(output, "[ERROR]")
}

func isCurrentUserRoot() bool {
	currentUser, err := user.Current()
	if err != nil {
		fmt.Println("Error getting current user:", err)
		os.Exit(1)
	}

	return currentUser.Uid == "0"
}

func joinInt64Slice(slice []int64, sep string) string {
	if len(slice) == 0 {
		return ""
	}
	result := fmt.Sprintf("%d", slice[0])
	for _, id := range slice[1:] {
		result += fmt.Sprintf("%s%d", sep, id)
	}
	return result
}
