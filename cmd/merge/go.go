package merge

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/spf13/cobra"
)

var GoCmd = &cobra.Command{
	Use:   "go",
	Short: "Run and debug federate feature",
	Run: func(cmd *cobra.Command, args []string) {
		debugFederate(manifest.Load())
	},
}

// 这里的内容可能经常变：通过运行来调试 federate 代码
func debugFederate(m *manifest.Manifest) {
	propertyManager := merge.NewPropertyManager(m)
	propertyManager.Silent().Debug().Analyze()
}

func init() {
	manifest.RequiredManifestFileFlag(GoCmd)
}
