package insight

import (
	"federate/internal/fs"
	"github.com/spf13/cobra"
)

var architectureCmd = &cobra.Command{
	Use:   "architecture",
	Short: "Display architecture layers",
	Run: func(cmd *cobra.Command, args []string) {
		runArchitectureCommand()
	},
}

func runArchitectureCommand() {
	fs.DisplayJPGInITerm2("templates/arch_layer.jpg")
}
