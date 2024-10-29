package optimize

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"regexp"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var dependencyCmd = &cobra.Command{
	Use:   "dependency",
	Short: "Optimize project JAR dependencies",
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.Load()
		optimizeDependency(manifest)
	},
}

type DependencyReport struct {
	UnusedDeclared []string
	UsedUndeclared []string
}

func optimizeDependency(m *manifest.Manifest) {
	cmd := exec.Command("mvn", "dependency:analyze")
	cmd.Dir = federated.GeneratedTargetRoot(m.Main.Name)

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	if err := cmd.Run(); err != nil {
		log.Fatalf("Error running mvn dependency:analyze: %v", err)
	}

	report := parseDependencyAnalysis(out.String())

	fmt.Println("Dependency Analysis Report:")

	if len(report.UsedUndeclared) > 0 {
		fmt.Println("\nUsed undeclared dependencies:")
		for _, dep := range report.UsedUndeclared {
			fmt.Printf("  - %s\n", dep)
		}
		fmt.Println("\nAction: Consider declaring these dependencies in your pom.xml")
	}

	if len(report.UnusedDeclared) > 0 {
		fmt.Println("\nUnused declared dependencies:")
		for _, dep := range report.UnusedDeclared {
			fmt.Printf("  - %s\n", dep)
		}
		fmt.Println("\nAction: Consider removing these dependencies from your pom.xml if they are truly unused")
	}

	if len(report.UsedUndeclared) == 0 && len(report.UnusedDeclared) == 0 {
		fmt.Println("No dependency issues found. Your project's dependencies appear to be well-managed.")
	} else {
		fmt.Println("\nSummary:")
		fmt.Printf("- Used undeclared dependencies: %d\n", len(report.UsedUndeclared))
		fmt.Printf("- Unused declared dependencies: %d\n", len(report.UnusedDeclared))
		fmt.Println("\nConsider reviewing and addressing the above dependency issues to improve your project structure.")
	}
}

func parseDependencyAnalysis(output string) DependencyReport {
	var report DependencyReport
	scanner := bufio.NewScanner(strings.NewReader(output))

	section := ""
	for scanner.Scan() {
		line := removeAnsiColors(scanner.Text())

		if strings.Contains(line, "Used undeclared dependencies found:") {
			section = "usedUndeclared"
		} else if strings.Contains(line, "Unused declared dependencies found:") {
			section = "unusedDeclared"
		} else if strings.Contains(line, "BUILD SUCCESS") {
			break // Stop parsing when we reach the build information
		} else if strings.TrimSpace(line) == "" || strings.Contains(line, "--------") {
			continue // Skip empty lines and separator lines
		} else if section == "usedUndeclared" {
			dep := strings.TrimSpace(line)
			dep = strings.TrimPrefix(dep, "[WARNING]")
			dep = strings.TrimPrefix(dep, "[INFO]")
			dep = strings.TrimSpace(dep)
			if dep != "" {
				report.UsedUndeclared = append(report.UsedUndeclared, dep)
			}
		} else if section == "unusedDeclared" {
			dep := strings.TrimSpace(line)
			dep = strings.TrimPrefix(dep, "[WARNING]")
			dep = strings.TrimPrefix(dep, "[INFO]")
			dep = strings.TrimSpace(dep)
			if dep != "" {
				report.UnusedDeclared = append(report.UnusedDeclared, dep)
			}
		}
	}

	return report
}

func removeAnsiColors(str string) string {
	ansi := regexp.MustCompile(`\x1b\[[0-9;]*m`)
	return ansi.ReplaceAllString(str, "")
}

func init() {
	manifest.RequiredManifestFileFlag(dependencyCmd)
}
