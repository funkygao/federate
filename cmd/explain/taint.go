package explain

import (
	"log"

	"github.com/spf13/cobra"
)

var taintCmd = &cobra.Command{
	Use:   "taint",
	Short: "Show resources that cannot be automatically merged",
	Long: `The taint command shows resources that cannot be automatically merged.

Example usage:
  federate microservice explain taint`,
	Run: func(cmd *cobra.Command, args []string) {
		showTaint()
	},
}

func showTaint() {
	log.Printf("taint: borrowed from Kubernetes")
	log.Printf("In federate, it means unreconcilable resources:")
	log.Printf("  federated.reconcile.taint.logConfigXml")
	log.Printf("  federated.reconcile.taint.mybatisConfigXml")
}
