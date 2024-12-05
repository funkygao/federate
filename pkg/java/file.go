package java

import (
	"os"
	"path/filepath"
	"strings"
)

func IsJavaMainSource(info os.FileInfo, path string) bool {
	return !info.IsDir() &&
		info.Size() > 20 &&
		strings.HasSuffix(info.Name(), ".java") &&
		!strings.HasSuffix(info.Name(), "package-info.java") &&
		!strings.Contains(path, "/target/") &&
		!strings.Contains(path, "/test/")
}

func IsXML(info os.FileInfo, path string) bool {
	return !info.IsDir() &&
		strings.HasSuffix(info.Name(), ".xml") &&
		!strings.Contains(path, "/target/") &&
		!strings.Contains(path, "/test/")
}

func IsResourceFile(info os.FileInfo, path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	_, exists := resourceExtensions[ext]
	return exists
}

func IsMetaInfFile(info os.FileInfo, path string) bool {
	return strings.Contains(filepath.ToSlash(path), "META-INF")
}

func IsSpringYamlFile(info os.FileInfo, path string) bool {
	base := filepath.Base(path)
	return yamlFilePattern.MatchString(base)
}
