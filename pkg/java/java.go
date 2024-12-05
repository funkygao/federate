package java

import (
	"path/filepath"
	"strings"
)

func ClassSimpleName(classFullName string) string {
	parts := strings.Split(classFullName, ".")
	return parts[len(parts)-1]
}

func ClassPackageName(classFullName string) string {
	parts := strings.Split(classFullName, ".")
	return strings.Join(parts[:len(parts)-1], ".")
}

func Pkg2Path(pkg string) string {
	return filepath.FromSlash(strings.ReplaceAll(pkg, ".", "/"))
}

func JavaFile2Class(javaSrcPath string) string {
	fileName := filepath.Base(javaSrcPath)
	return strings.TrimSuffix(fileName, ".java")
}
