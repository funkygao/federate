package java

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	resourceExtensions = map[string]struct{}{
		".json":       {}, // private ConfigMaps
		".xml":        {}, // beans, mybatis-config
		".properties": {}, // i18N message bundles, properties
		".yml":        {}, // spring boot
		".html":       {}, // email templates
	}

	// 使用不区分大小写的正则表达式来匹配 Spring 配置文件
	yamlFilePattern = regexp.MustCompile(`(?i)^application(-[a-zA-Z0-9\-]+)?\.(yml)$`)
)

func ClassSimpleName(classFullName string) string {
	parts := strings.Split(classFullName, ".")
	return parts[len(parts)-1]
}

func ClassPackageName(classFullName string) string {
	parts := strings.Split(classFullName, ".")
	return strings.Join(parts[:len(parts)-1], ".")
}

func IsJavaMainSource(info os.FileInfo, path string) bool {
	return !info.IsDir() &&
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

func Pkg2Path(pkg string) string {
	return filepath.FromSlash(strings.ReplaceAll(pkg, ".", "/"))
}

func JavaFile2Class(javaSrcPath string) string {
	fileName := filepath.Base(javaSrcPath)
	return strings.TrimSuffix(fileName, ".java")
}
