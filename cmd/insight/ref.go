package insight

import (
	"log"

	"federate/pkg/javast"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var refCmd = &cobra.Command{
	Use:   "ref <dir> FQCN[#propertyName]",
	Short: "Trace Class|Property references from a Java source directory",
	Long: `The ref command trace Class|Property references from a Java source directory.

Example usage:
  federate insight ref . com.jdwl.wms.shipment.order.domain.order.entity.ShipmentOrder
  federate insight ref . com.jdwl.wms.shipment.order.domain.order.entity.ShipmentOrder#orderType`,
	Args: cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		runRefCommand(args[0], args[1])
	},
}

func runRefCommand(root, name string) {
	driver := javast.NewJavastDriver().Verbose()
	info, err := driver.ExtractRef(root, name)
	if err != nil {
		log.Fatalf("%s %v", name, err)
	}

	log.Println(util.Beautify(info))
}
