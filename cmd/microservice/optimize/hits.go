package optimize

import (
	"federate/pkg/javast"
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var hitsTopK int

var hitsCmd = &cobra.Command{
	Use:   "hits",
	Short: "Analyze HITS: Hyperlink-Induced Topic Search",
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.Load()
		analyzeHITS(manifest)
	},
}

func analyzeHITS(m *manifest.Manifest) {
	for _, c := range m.Components {
		javast.AnalyzeHITS(c, hitsTopK)
	}
}

func init() {
	manifest.RequiredManifestFileFlag(hitsCmd)
	hitsCmd.Flags().IntVarP(&hitsTopK, "topk", "k", 20, "top k")
}
