package plus

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "plus",
	Short: "Develop new and tailor existing capabilities atop WMS6.0",
	Long: `Create new functionalities and adaptations leveraging WMS6.0, without modifying the core platform.
Enables custom solutions through extension points and other extensibility mechanisms while preserving WMS6.0 integrity.

  Find more information at: http://w6-developer.jdl.com/`,
}

func init() {
	CmdGroup.AddCommand(createCmd, manifestCmd)
}
