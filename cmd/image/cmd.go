package image

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "image",
	Short: "Commands for managing images like Docker and RPM",
	Long:  `The image command group provides a set of commands to manage images like Docker and RPM.`,
}

func init() {
	CmdGroup.AddCommand(buildRpmCmd, buildDockerCmd)
}
