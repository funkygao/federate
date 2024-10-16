package optimize

import (
	"bufio"
	"fmt"
	"os"

	"federate/pkg/manifest"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	autoYes bool
)

var CmdGroup = &cobra.Command{
	Use:   "optimize",
	Short: "Identify potential areas for optimization",
	Long:  `The optimize command identify potential areas for optimization.`,
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.LoadManifest()
		optimize(manifest)
	},
}

func optimize(m *manifest.Manifest) {
	steps := []struct {
		name string
		fn   func()
	}{
		{"Detect similar classes for potential duplicate coding", func() {
			showDuplicates(m)
		}},
		{"Optimize project JAR dependencies", func() {
		}},
	}

	totalSteps := len(steps)
	for i, step := range steps {
		promptToProceed(i+1, totalSteps, step.name)
		step.fn()
	}
}

func promptToProceed(seq, total int, step string) {
	c := color.New(color.FgMagenta)
	c.Printf("Step [%d/%d] %s ...", seq, total, step)
	if autoYes {
		fmt.Println()
		return
	}
	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')
}

func init() {
	CmdGroup.AddCommand(duplicateCmd, jarCmd)

	manifest.RequiredManifestFileFlag(CmdGroup)
}
