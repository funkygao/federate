package workload

import (
	"github.com/spf13/cobra"
)

var integrationCmd = &cobra.Command{
	Use:   "integration",
	Short: "Integration with workload SDK",
	Run: func(cmd *cobra.Command, args []string) {
	},
}
