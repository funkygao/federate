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

	"federate/pkg/javast/api"
	"federate/pkg/javast/ast"
	"federate/pkg/util"
)

func (d *javastDriver) ExtractRef(root, name string) (map[string][]string, error) {
	var info map[string][]string
	if err := d.extractData(root, CmdExtractRef, &info, name); err != nil {
		return nil, err
	}
	return info, nil
}

func (d *javastDriver) ExtractAST(root string) (*ast.Info, error) {
	var info ast.Info
	if err := d.extractData(root, CmdExtractAST, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

func (d *javastDriver) ExtractAPI(root string) (*api.Info, error) {
	var info api.Info
	if err := d.extractData(root, CmdExtractAPI, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

func (d *javastDriver) extractData(root string, cmd CmdName, result interface{}, extra ...string) error {
	tempDir, jarPath, err := prepareJavastJar()
	if err != nil {
		return err
	}
	defer os.RemoveAll(tempDir)

	args := []string{"-jar", jarPath, root, string(cmd)}
	args = append(args, extra...)
	execCmd := exec.Command("java", args...)
	if d.verbose {
		log.Printf("[%s] Executing: %s", root, strings.Join(execCmd.Args, " "))
	}
	var stdout, stderr bytes.Buffer
	execCmd.Stdout = &stdout
	execCmd.Stderr = &stderr

	t0 := time.Now()

	err = execCmd.Run()
	if err != nil {
		return fmt.Errorf("error running Extractor: %v\nStderr: %s", err, stderr.String())
	}

	t1 := time.Now()

	b := stdout.Bytes()
	if err = json.Unmarshal(b, result); err != nil {
		return fmt.Errorf("error parsing JSON output: %v", err)
	}

	if d.verbose {
		log.Printf("Java Extract cost: %s, JSON Unmarshal cost: %s, JSON Size: %s",
			t1.Sub(t0), time.Since(t1), util.ByteSize(len(b)))
	}

	return nil
}
