package api

import (
	"strings"
)

func isJDKType(typeName string) bool {
	// 处理数组类型
	if strings.HasSuffix(typeName, "[]") {
		return isJDKType(strings.TrimSuffix(typeName, "[]"))
	}

	// 处理泛型类型
	if idx := strings.Index(typeName, "<"); idx != -1 {
		baseType := typeName[:idx]
		return isJDKType(baseType)
	}

	// 基本类型
	primitiveTypes := map[string]bool{
		"boolean": true,
		"byte":    true,
		"char":    true,
		"short":   true,
		"int":     true,
		"long":    true,
		"float":   true,
		"double":  true,
		"void":    true,
	}
	if primitiveTypes[typeName] {
		return true
	}

	// 常用 JDK 包
	jdkPackages := []string{
		"java.lang.",
		"java.util.",
		"java.io.",
		"java.nio.",
		"java.time.",
		"java.math.",
		"java.net.",
		"java.text.",
		"java.sql.",
		"javax.sql.",
	}

	for _, pkg := range jdkPackages {
		if strings.HasPrefix(typeName, pkg) {
			return true
		}
	}

	// 常用 JDK 类（不带包名的情况）
	commonJDKClasses := map[string]bool{
		"String":           true,
		"Object":           true,
		"Class":            true,
		"Enum":             true,
		"System":           true,
		"Thread":           true,
		"Runnable":         true,
		"Throwable":        true,
		"Exception":        true,
		"RuntimeException": true,
		"Error":            true,
		"Integer":          true,
		"Long":             true,
		"Double":           true,
		"Float":            true,
		"Boolean":          true,
		"Byte":             true,
		"Character":        true,
		"Short":            true,
		"Void":             true,
		"BigDecimal":       true,
		"BigInteger":       true,
		"StringBuilder":    true,
		"StringBuffer":     true,
		"List":             true,
		"Map":              true,
		"Set":              true,
		"Collection":       true,
		"Iterator":         true,
		"Stream":           true,
		"Optional":         true,
		"Arrays":           true,
		"Collections":      true,
		"Date":             true,
		"Calendar":         true,
		"LocalDate":        true,
		"LocalTime":        true,
		"LocalDateTime":    true,
		"Instant":          true,
		"File":             true,
		"Path":             true,
		"Files":            true,
		"InputStream":      true,
		"OutputStream":     true,
		"Reader":           true,
		"Writer":           true,
	}

	return commonJDKClasses[typeName]
}
