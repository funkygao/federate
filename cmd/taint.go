package cmd

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
	log.Printf("log_config_xml")
	log.Printf("mybatis_config_xml")
}
