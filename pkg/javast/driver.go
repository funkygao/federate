package javast

import (
	"bytes"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/manifest"
)

type JavastDriver interface {
	Invoke(command string, jsonArgs string) error
}

type javastDriver struct {
	c manifest.ComponentInfo
}

func NewJavastDriver(c manifest.ComponentInfo) JavastDriver {
	return &javastDriver{c: c}
}

func (d *javastDriver) Invoke(command, jsonArgs string) error {
	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return err
	}
	defer os.RemoveAll(tempDir)

	cmd := exec.Command("java", "-jar", jarPath, command, d.c.RootDir(), jsonArgs)
	log.Printf("[%s] Executing: %s", d.c.Name, strings.Join(cmd.Args, " "))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		log.Printf("Error: %s", stderr.String())
	}
	out := stdout.String()
	if out != "" {
		log.Println(out)
	}
	return err
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
