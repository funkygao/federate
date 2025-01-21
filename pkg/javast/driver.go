package javast

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"federate/internal/fs"
	"federate/pkg/javast/api"
	"federate/pkg/javast/ast"
	"federate/pkg/step"
)

type CmdName string

const (
	CmdReplaceService          CmdName = "replace-service"
	CmdInjectTrxManager        CmdName = "inject-transaction-manager"
	CmdUpdatePropertyRefKey    CmdName = "update-property-keys"
	CmdTransformImportResource CmdName = "transform-import-resource"
	CmdTransformResourceInject CmdName = "transform-resource"
	CmdExtractAST              CmdName = "extract-ast"
	CmdExtractAPI              CmdName = "extract-api"
	CmdExtractRef              CmdName = "extract-ref"
)

type Command struct {
	Name    CmdName
	RootDir string
	Args    string
}

type JavastDriver interface {
	Invoke(bar step.Bar, commands ...Command) error
	ExtractAST(root string) (*ast.Info, error)
	ExtractAPI(root string) (*api.Info, error)
	ExtractRef(root, name string) (map[string][]string, error)
	Verbose() JavastDriver
}

type javastDriver struct {
	verbose bool
}

func NewJavastDriver() JavastDriver {
	return &javastDriver{verbose: false}
}

func (d *javastDriver) Verbose() JavastDriver {
	d.verbose = true
	return d
}

func (d *javastDriver) Invoke(bar step.Bar, commands ...Command) error {
	oldBarDesc := bar.State().Description
	defer bar.Describe(oldBarDesc)

	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return err
	}
	defer os.RemoveAll(tempDir)

	bar.Describe(fmt.Sprintf("%s Grouping Components for Batch", oldBarDesc))
	groupedCommands := d.groupByRootDir(commands...)
	bar.Add(2)

	for rootDir, cmds := range groupedCommands {
		args := []string{"-jar", jarPath, rootDir}
		for _, cmd := range cmds {
			args = append(args, string(cmd.Name), cmd.Args)
		}

		cmd := exec.Command("java", args...)
		if d.verbose {
			log.Printf("[%s] Executing: %s", rootDir, strings.Join(cmd.Args, " "))
		}

		bar.Describe(fmt.Sprintf("%s Transforming %s/", oldBarDesc, rootDir))

		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr

		err = cmd.Run()
		if err != nil {
			log.Printf("Error: %s", stderr.String())
			return err
		}

		bar.Add(98 / len(groupedCommands))

		// 在进度条更新后输出 Java 进程的结果
		out := strings.TrimSpace(stdout.String())
		if out != "" {
			log.Println()
			log.Println(out)
		}
	}

	return nil
}

func (d *javastDriver) groupByRootDir(commands ...Command) map[string][]Command {
	grouped := make(map[string][]Command)
	for _, cmd := range commands {
		grouped[cmd.RootDir] = append(grouped[cmd.RootDir], cmd)
	}
	return grouped
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
