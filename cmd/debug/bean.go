package debug

import (
	"log"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/spring"
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
	springXmlPath := filepath.Join(m.TargetResourceDir(), "federated/spring.xml")
	manager := spring.New()
	ok, path := manager.SearchBean(springXmlPath, beanId)
	if ok {
		log.Printf("bean[%s] found in %s", beanId, path)
	} else {
		log.Printf("bean[%s] not found", beanId)
	}

}

func init() {
	manifest.RequiredManifestFileFlag(beanCmd)
}
