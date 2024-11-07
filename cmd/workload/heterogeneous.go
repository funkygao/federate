package workload

import (
	"github.com/spf13/cobra"
)

var heterogeneousCmd = &cobra.Command{
	Use:   "heterogeneous",
	Short: "Boost processing of large payload requests using GPU",
	Run: func(cmd *cobra.Command, args []string) {
	},
}
