package util

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"federate/internal/fs"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	tla2ToolsPath    string
	maxHeapSize      int
	invariants       []string
	collectCoverage  bool
	simulateMode     bool
	verboseMode      bool
	generateDemoMode bool
)

var tlaplusCmd = &cobra.Command{
	Use:   "tlaplus <.tla file>",
	Short: "Run TLA+ specification",
	Long: `This command runs a TLA+ specification file using the TLC model checker.

Make sure you have the TLA+ tools installed and the 'tla2tools.jar' file is accessible.
Download from: https://github.com/tlaplus/tlaplus/releases/latest`,
	Args: cobra.RangeArgs(0, 1),
	Run: func(cmd *cobra.Command, args []string) {
		if generateDemoMode {
			generateDemo()
		} else {
			if len(args) != 1 {
				cmd.Help()
				return
			}
			useTLAplus(args[0])
		}
	},
}

func useTLAplus(specFile string) {
	if filepath.Ext(specFile) != ".tla" {
		fmt.Println("Error: The specified file must have a .tla extension")
		os.Exit(1)
	}

	// 初始化 Java 参数
	javaArgs := []string{fmt.Sprintf("-Xmx%dg", maxHeapSize)}

	// 处理 classpath
	sysClasspath := os.Getenv("CLASSPATH")
	var classpath string
	if tla2ToolsPath != "tla2tools.jar" {
		// 如果用户明确指定了 tla2tools.jar 的路径
		classpath = tla2ToolsPath
	} else if sysClasspath != "" {
		// 如果没有明确指定，但系统 CLASSPATH 不为空
		// 在系统 CLASSPATH 中查找 tla2tools.jar
		paths := strings.Split(sysClasspath, ":")
		for _, path := range paths {
			if _, err := os.Stat(filepath.Join(path, "tla2tools.jar")); err == nil {
				classpath = filepath.Join(path, "tla2tools.jar")
				break
			}
		}
		if classpath == "" {
			fmt.Println("Error: tla2tools.jar not found in CLASSPATH")
			os.Exit(1)
		}
	} else {
		// 如果既没有明确指定，系统 CLASSPATH 也为空，则使用当前目录
		if _, err := os.Stat("tla2tools.jar"); err == nil {
			classpath = "tla2tools.jar"
		} else {
			fmt.Println("Error: tla2tools.jar not found in current directory")
			os.Exit(1)
		}
	}
	javaArgs = append(javaArgs, "-cp", classpath)

	// 添加 TLC 主类和规范文件
	javaArgs = append(javaArgs, "tlc2.TLC")

	// 检查配置文件是否存在
	cfgFile := strings.TrimSuffix(specFile, ".tla") + ".cfg"
	if _, err := os.Stat(cfgFile); err == nil {
		javaArgs = append(javaArgs, "-config", cfgFile)
	}

	for _, inv := range invariants {
		javaArgs = append(javaArgs, "-invariant", inv)
	}
	if collectCoverage {
		javaArgs = append(javaArgs, "-coverage", "1")
	}
	if simulateMode {
		javaArgs = append(javaArgs, "-simulate")
	}
	if verboseMode {
		javaArgs = append(javaArgs, "-debug")
	}

	javaArgs = append(javaArgs, "-workers", fmt.Sprintf("%v", runtime.NumCPU()))
	javaArgs = append(javaArgs, specFile)
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

func generateDemo() {
	files := []string{"SimpleCounter.tla", "SimpleCounter.cfg"}
	for _, f := range files {
		log.Printf("Generated: %s", f)
		fs.GenerateFileFromTmpl(filepath.Join("templates", "tla", f), f, nil)
	}
}

func init() {
	tlaplusCmd.Flags().StringVarP(&tla2ToolsPath, "tla2tools", "t", "tla2tools.jar", "Path to tla2tools.jar (default: use system CLASSPATH)")
	tlaplusCmd.Flags().IntVarP(&maxHeapSize, "max-heap", "m", 75, "Maximum heap size in GB")
	tlaplusCmd.Flags().StringSliceVarP(&invariants, "invariant", "i", []string{}, "Invariants to check")
	tlaplusCmd.Flags().BoolVarP(&collectCoverage, "coverage", "c", false, "Collect coverage information")
	tlaplusCmd.Flags().BoolVarP(&verboseMode, "verbose", "v", false, "Verbose mode")
	tlaplusCmd.Flags().BoolVarP(&simulateMode, "simulate", "s", false, "Run in simulation mode")
	tlaplusCmd.Flags().BoolVarP(&generateDemoMode, "generate-demo", "g", false, "Generate tla specification files")
}
