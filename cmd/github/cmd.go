package github

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "github",
	Short: "GitHub commands for exploration and discovery",
}

func init() {
	CmdGroup.AddCommand(topReposCmd, languagesCmd)
}
