package snap

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
	"federate/pkg/step"
	"github.com/spf13/cobra"
)

var Cmd = &cobra.Command{
	Use:   "snap",
	Short: "Create a secure, tailored source code snapshot for customer delivery",
	Long: `Snap creates a tailored, secure source code snapshot ready for customer delivery.

Key features:
1. Processes git submodules in-place, enabling easy review via diff
2. Removes sensitive information and unnecessary code
3. Eliminates configuration files unrelated to the target profile
4. Ensures post-delivery compilability, allowing customers to build the code independently

The command balances transparency with proprietary protection by:
- Sanitizing internal references and configurations
- Preparing a local Maven repository with necessary dependencies
- Adjusting build scripts for customer environment compatibility
- Retaining only relevant configuration for the specified profile

Outputs:
1. Source Code Repository:
   A git repository tailored for the target environment, with sensitive
   information and irrelevant configurations removed.

2. Local Maven Repository:
   A collection of JAR files and dependencies required for compilation,
   without exposing proprietary source code.

These outputs form a professional-grade, ready-to-deliver package that
enables customers to build and run the application independently.`,
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		runSnap(m)
	},
}

var (
	autoYes           bool
	enableObfuscation bool
	hinted            = false
	absLocalRepoPath  string
	relativeRepoPath  string
)

func runSnap(m *manifest.Manifest) {
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current working directory: %v", err)
	}

	// 设置 absLocalRepoPath 为绝对路径
	relativeRepoPath = filepath.Join("generated", "artifactory")
	absLocalRepoPath = filepath.Join(currentDir, relativeRepoPath)

	steps := []step.Step{
		{
			Name: fmt.Sprintf("Create local Maven repository: %s/", relativeRepoPath),
			Fn:   createLocalMavenRepo,
		},
		{
			Name: "Update component pom.xml to use the local Maven repository",
			Fn:   func(bar step.Bar) { updatePomFilesForLocalRepo(m) },
		},
		{
			Name: fmt.Sprintf("Copy dependency JAR to local Maven repository: %s/", relativeRepoPath),
			Fn:   func(bar step.Bar) { copyDependenciesToLocalRepo(m, bar) },
		},
		{
			Name: "Obfuscate local Maven repository JAR",
			Fn:   func(bar step.Bar) { obfuscateJars(m, bar) },
		},
		{
			Name: "Detect JAR version conflicts",
			Fn:   detectVersionConflicts,
		},
		{
			Name: fmt.Sprintf("Reorganize %s to follow the standard Maven repository layout", relativeRepoPath),
			Fn:   organizeLocalRepo,
		},
		{
			Name: fmt.Sprintf("Sanitize Code Repo: %s", m.Main.Name),
			Fn:   func(bar step.Bar) { sanitizeCodeRepo(m) },
		},
		{
			Name: "Final check",
			Fn:   func(bar step.Bar) { finalChecks(0) },
		},
	}

	step.AutoConfirm = autoYes
	step.Run(steps)
	fmt.Println()

	ledger.Get().SaveToFile("report.json")
}

func init() {
	manifest.RequiredManifestFileFlag(Cmd)
	Cmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	Cmd.Flags().BoolVar(&enableObfuscation, "obfuscate", true, "Enable JAR obfuscation")
}
