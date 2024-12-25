package federated

import (
	"path/filepath"
	"strings"
)

const (
	GeneratedDir = ""
	FederatedDir = "federated"

	starterSuffix = "-starter"
)

var resourceDir = filepath.Join("src", "main", "resources", FederatedDir)

func GeneratedResourceBaseDir(targetSystemName string) string {
	return filepath.Join(GeneratedDir, targetSystemName, resourceDir)
}

func ResourceBaseName(filePath string) string {
	parts := strings.SplitN(filePath, FederatedDir+string(filepath.Separator), 2)
	if len(parts) == 2 {
		return parts[1]
	}
	return filePath // 如果没有找到 "federated/"，则使用原始路径
}

func GeneratedTargetRoot(targetSystemName string) string {
	return filepath.Join(GeneratedDir, targetSystemName)
}

func StarterBaseDir(targetSystemName string) string {
	return targetSystemName + starterSuffix
}
