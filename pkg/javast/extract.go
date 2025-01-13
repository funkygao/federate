package javast

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"federate/pkg/javast/ast"
)

// ASTInfo 结构体用于解析 JSON 输出
type ASTInfo struct {
	Imports     []string `json:"imports"`
	Classes     []string `json:"classes"`
	Methods     []string `json:"methods"`
	Variables   []string `json:"variables"`
	MethodCalls []string `json:"methodCalls"`
}

func (d *javastDriver) ExtractAST(root string) (*ast.Info, error) {
	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	args := []string{"-jar", jarPath, root, "extract-ast"}
	cmd := exec.Command("java", args...)
	if d.verbose {
		log.Printf("[%s] Executing: %s", root, strings.Join(cmd.Args, " "))
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("error running ASTExtractor: %v\nStderr: %s", err, stderr.String())
	}

	var astInfo ast.Info
	err = json.Unmarshal(stdout.Bytes(), &astInfo)
	if err != nil {
		return nil, fmt.Errorf("error parsing JSON output: %v", err)
	}

	return &astInfo, nil
}
