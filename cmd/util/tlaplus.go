package util

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	tla2ToolsPath string
)

var tlaplusCmd = &cobra.Command{
	Use:   "tlaplus <.tla file>",
	Short: "Run TLA+ specification",
	Long: `This command runs a TLA+ specification file using the TLC model checker.

Make sure you have the TLA+ tools installed and the 'tla2tools.jar' file is accessible.
Download from: https://github.com/tlaplus/tlaplus/releases/latest`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		useTLAplus(args[0])
	},
}

func useTLAplus(file string) {
	if filepath.Ext(file) != ".tla" {
		fmt.Println("Error: The specified file must have a .tla extension")
		os.Exit(1)
	}

	javaArgs := []string{"-cp", tla2ToolsPath, "tlc2.TLC", file}

	cmd := exec.Command("java", javaArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	color.Cyan("Executing: %s", strings.Join(cmd.Args, " "))
	err := cmd.Run()
	if err != nil {
		fmt.Printf("Error running TLA+ specification: %v\n", err)
		os.Exit(1)
	}
}

func init() {
	tlaplusCmd.Flags().StringVarP(&tla2ToolsPath, "tla2tools", "t", "tla2tools.jar", "Path to tla2tools.jar")
}
