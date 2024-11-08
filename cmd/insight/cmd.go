package insight

import (
	"github.com/spf13/cobra"
)

var debug bool

var CmdGroup = &cobra.Command{
	Use:   "insight",
	Short: "Gain insights from Java source code",
	Long:  `The insight command group provides tools to analyze Java source code and extract various metrics and information.`,
}

func init() {
	CmdGroup.AddCommand(methodsCmd)
}
