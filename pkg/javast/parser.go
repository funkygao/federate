package javast

import (
	"encoding/json"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/internal/fs"
)

func RecursiveParse(command, rootDir string) ([]map[string]interface{}, error) {
	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	cmd := exec.Command("java", "-jar", jarPath, command, rootDir)
	log.Printf("%s", strings.Join(cmd.Args, " "))
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	// stdout is jar execution output
	var results []map[string]interface{}
	err = json.Unmarshal(output, &results)
	if err != nil {
		return nil, err
	}

	return results, nil
}

func prepareJavastJar() (string, string, error) {
	jarContent, err := fs.FS.ReadFile("templates/jar/javast.jar")
	if err != nil {
		return "", "", err
	}

	// 创建临时目录
	tempDir, err := os.MkdirTemp("", "javast")
	if err != nil {
		return "", "", err
	}

	// 创建临时 JAR 文件
	jarPath := filepath.Join(tempDir, "javast.jar")
	err = os.WriteFile(jarPath, jarContent, 0644)
	if err != nil {
		os.RemoveAll(tempDir) // 清理临时目录
		return "", "", err
	}

	return tempDir, jarPath, nil
}
