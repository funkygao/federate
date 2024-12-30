package debug

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "debug",
	Short: "Debug tools for fusion project development",
}

func init() {
	CmdGroup.AddCommand(beanCmd, refCmd, ymlCmd, pomCmd, wizardCmd)
}
