package debug

import (
	"log"

	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var beanCmd = &cobra.Command{
	Use:   "bean <id>",
	Short: "Search bean with id from federated/spring.xml",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		searchBean(m, args[0])
	},
}

func searchBean(m *manifest.Manifest, beanId string) {
	log.Printf("bean[%s]", beanId)
}

func init() {
	manifest.RequiredManifestFileFlag(beanCmd)
}
