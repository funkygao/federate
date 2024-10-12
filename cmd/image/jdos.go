package image

import (
	"github.com/spf13/cobra"
)

var jdosCmd = &cobra.Command{
	Use:   "jdos",
	Short: "Build and prepare application for deployment on JDOS 3.0",
	Run: func(cmd *cobra.Command, args []string) {
		buildForJDOS()
	},
}

func buildForJDOS() {
}
