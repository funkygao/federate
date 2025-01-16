package javast

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"federate/pkg/javast/ast"
	"federate/pkg/util"
)

func (d *javastDriver) ExtractAST(root string) (*ast.Info, error) {
	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	args := []string{"-jar", jarPath, root, string(CmdExtractAST)}
	cmd := exec.Command("java", args...)
	if d.verbose {
		log.Printf("[%s] Executing: %s", root, strings.Join(cmd.Args, " "))
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	t0 := time.Now()

	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("error running ASTExtractor: %v\nStderr: %s", err, stderr.String())
	}

	t1 := time.Now()

	var astInfo ast.Info
	b := stdout.Bytes()
	if err = json.Unmarshal(b, &astInfo); err != nil {
		return nil, fmt.Errorf("error parsing JSON output: %v", err)
	}

	if d.verbose {
		log.Printf("Java AST Extract cost: %s, AST JSON Unmarshal cost: %s, JSON Size: %s",
			t1.Sub(t0), time.Since(t1), util.ByteSize(len(b)))
	}

	return &astInfo, nil
}
