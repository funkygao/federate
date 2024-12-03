package microservice

import (
	"io"
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge"
	"federate/pkg/merge/ledger"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

const (
	defaultCellMaxWidth = 40
	targetSpringXml     = "spring.xml"
)

var (
	yamlConflictCellMaxWidth int
	dryRunMerge              bool
	autoYes                  bool
	silentMode               bool
	noColor                  bool
)

var mergeCmd = &cobra.Command{
	Use:   "consolidate",
	Short: "Merge components into target system following directives of the manifest",
	Long: `The merge command merges components into target system following directives of the manifest.

  See: https://mwhittaker.github.io/publications/service_weaver_HotOS2023.pdf

Example usage:
  federate microservice consolidate -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		doMerge(manifest.Load())
	},
}

func doMerge(m *manifest.Manifest) {
	if silentMode {
		log.SetOutput(io.Discard)
	}
	if noColor {
		color.NoColor = true
	}

	compiler := merge.NewCompiler(m)
	compiler.WithOption(merge.WithDryRun(dryRunMerge))
	compiler.WithOption(merge.WithAutoYes(autoYes))
	compiler.Init().Merge()

	ledger.Get().ShowSummary()
}

func init() {
	manifest.RequiredManifestFileFlag(mergeCmd)
	mergeCmd.Flags().BoolVarP(&autoYes, "yes", "y", false, "Automatically answer yes to all prompts")
	mergeCmd.Flags().BoolVarP(&silentMode, "silent", "s", false, "Silent or quiet mode")
	mergeCmd.Flags().IntVarP(&yamlConflictCellMaxWidth, "yaml-conflict-cell-width", "w", defaultCellMaxWidth, "Yml files conflict table cell width")
	mergeCmd.Flags().BoolVarP(&dryRunMerge, "dry-run", "d", false, "Perform a dry run without making any changes")
	mergeCmd.Flags().BoolVarP(&noColor, "no-color", "n", false, "Disable colorized output")
	mergeCmd.Flags().BoolVarP(&merge.FailFast, "fail-fast", "f", false, "Fail fast")
}
