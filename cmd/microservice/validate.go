package microservice

import (
	"log"

	"federate/pkg/compliance"
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var validateCmd = &cobra.Command{
	Use:   "validate",
	Short: "Check components for compliance with development conventions",
	Long: `The validate command verifies that the components specified in the manifest file
comply with the development conventions.

Example usage:
  federate microservice validate -i manifest.yaml`,
	Run: func(cmd *cobra.Command, args []string) {
		m, err := manifest.LoadManifest()
		if err != nil {
			log.Fatalf("Error loading manifest: %v", err)
		}

		compliance.CheckComponentsCompliance(m)
	},
}

func init() {
	manifest.RequiredManifestFileFlag(validateCmd)
}
