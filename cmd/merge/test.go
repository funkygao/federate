package merge

import (
	"federate/pkg/manifest"
	"federate/pkg/merge"
	"github.com/spf13/cobra"
)

// TestCmd is just for merge debugging, will be killed on release.
var TestCmd = &cobra.Command{
	Use:   "test",
	Short: "Test temporary",
	Run: func(cmd *cobra.Command, args []string) {
		runTest(manifest.Load())
	},
}

// 这里的内容可能经常变：通过运行来调试 federate 代码
func runTest(m *manifest.Manifest) {
	propertyManager := merge.NewPropertyManager(m)
	identifyPropertyConflicts(m, propertyManager)
}

func init() {
	manifest.RequiredManifestFileFlag(TestCmd)
}
