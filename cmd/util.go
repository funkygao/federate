package cmd

import (
	"github.com/spf13/cobra"
)

// addRequiredInputFlag adds the --input flag to the given command and marks it as required
func addRequiredInputFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&manifestFile, "input", "i", "", "Path to the manifest file")
	cmd.MarkFlagRequired("input")
}
