package github

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "github",
	Short: "Commands for exploring GitHub trends and popular repositories",
}

func init() {
	CmdGroup.AddCommand(topReposCmd, languagesCmd)
}
