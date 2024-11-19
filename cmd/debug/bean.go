package debug

import (
	"federate/pkg/manifest"
	"federate/pkg/spring"
	"github.com/fatih/color"
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
	manager := spring.New(true)
	for _, bean := range manager.ListBeans(m.SpringXmlPath(), spring.QueryBeanID(beanId)) {
		if bean.Value == beanId {
			color.Green("bean[%s] found in %s", beanId, bean.FileName)
			return
		}
	}
	color.Red("bean[%s] not found", beanId)

}

func init() {
	manifest.RequiredManifestFileFlag(beanCmd)
}
