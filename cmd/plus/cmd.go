package plus

import (
	"federate/cmd/plus/snap"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "wms-plus",
	Short: "Develop new and tailor capabilities atop WMS6",
	Long: `Create new functionalities and adaptations leveraging WMS6, without modifying the core platform.

  Find more information at: http://w6-developer.jdl.com/`,
}

func init() {
	CmdGroup.AddCommand(scaffoldCmd, snap.Cmd, manifestCmd)
}
