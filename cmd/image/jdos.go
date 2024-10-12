package image

import (
	"github.com/spf13/cobra"
)

var jdosCmd = &cobra.Command{
	Use:   "build-jdos",
	Short: "Build target system for deployment on JDOS 3.0",
	Run: func(cmd *cobra.Command, args []string) {
		buildForJDOS()
	},
}

func buildForJDOS() {
}
