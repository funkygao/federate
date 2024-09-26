package cmd

import (
	"log"

	"github.com/spf13/cobra"
)

var projectName string

var scaffoldCmd = &cobra.Command{
	Use:   "create",
	Short: "Scaffold a new federated target system",
	Long: `The create command scaffolds a new federated target system.

Example usage:
  federate microservice create --name foo`,
	Run: func(cmd *cobra.Command, args []string) {
		scaffoldProject()
	},
}

func scaffoldProject() {
	if projectName == "" {
		log.Fatalf("--name is required")
	}
}

func init() {
	scaffoldCmd.Flags().StringVarP(&projectName, "name", "n", "", "Target system name")
}
