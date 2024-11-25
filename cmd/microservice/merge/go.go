package merge

import (
	"log"

	"federate/pkg/manifest"
	"federate/pkg/merge/property"
	"federate/pkg/merge/transformer"
	"github.com/spf13/cobra"
)

var logFlag int

var GoCmd = &cobra.Command{
	Use:   "go",
	Short: "Run and debug federate feature",
	Run: func(cmd *cobra.Command, args []string) {
		debugFederate(manifest.Load())
	},
}

// 这里的内容可能经常变：通过运行来调试 federate 代码
func debugFederate(m *manifest.Manifest) {
	switch logFlag {
	case 2:
		log.SetFlags(log.Lshortfile)
	case 3:
		log.SetFlags(log.Llongfile)
	}

	manager := property.NewManager(m)
	manager.Silent().Debug().Analyze()
	reconcilePropertiesConflicts(manager)
	transformer.Get().ShowSummary()
}

func init() {
	manifest.RequiredManifestFileFlag(GoCmd)
	GoCmd.Flags().IntVarP(&logFlag, "log", "l", 1, "2: Lshortfile, 3: Lshortfile")
}
