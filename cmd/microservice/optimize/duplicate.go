package optimize

import (
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/similarity"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	similarityThreshold float64
	similarityAlgo      string
	optimizeVerbosity   int
)

var duplicateCmd = &cobra.Command{
	Use:   "duplicate",
	Short: "Identify potential duplicate classes",
	Run: func(cmd *cobra.Command, args []string) {
		manifest := manifest.Load()
		showDuplicates(manifest)
	},
}

func showDuplicates(m *manifest.Manifest) {
	detector := similarity.NewDetector(m, similarityThreshold, similarityAlgo)
	pairs, err := detector.Detect()
	if err != nil {
		log.Fatalf("%v", err)
	}

	// 首先按 File1 排序，然后按 Similarity 降序排序
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].File1 != pairs[j].File1 {
			return pairs[i].File1 < pairs[j].File1
		}
		return pairs[i].Similarity > pairs[j].Similarity
	})

	// 创建一个map来存储每个文件的相似文件
	similarFiles := make(map[string][]similarity.DuplicatePair)

	for _, pair := range pairs {
		similarFiles[pair.File1] = append(similarFiles[pair.File1], pair)
		// 不再需要为 File2 添加条目，因为我们现在按 File1 排序
	}

	// 显示结果
	for file, pairs := range similarFiles {
		color.Yellow(file)
		for _, pair := range pairs {
			log.Printf("  - %.2f%% %s\n", pair.Similarity*100, pair.File2)
		}
		log.Println()
	}

	log.Printf("Total: %d unique files with similarities detected", len(similarFiles))
}

func init() {
	manifest.RequiredManifestFileFlag(duplicateCmd)
	duplicateCmd.Flags().Float64VarP(&similarityThreshold, "similarity-threshold", "t", 0.91, "Similarity threshold (0.0 - 1.0)")
	duplicateCmd.Flags().IntVarP(&optimizeVerbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
	duplicateCmd.Flags().StringVarP(&similarityAlgo, "similarity-algorithm", "s", "simhash", "simhash | jaccard")
}
