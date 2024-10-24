package optimize

import (
	"fmt"
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/optimizer"
	"github.com/spf13/cobra"
)

var (
	similarityThreshold float64
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
	dups, err := optimizer.DetectDuplicateJava(m)
	if err != nil {
		log.Fatalf("Error detecting duplicate java: %v", err)
	}

	if len(dups) == 0 {
		log.Println("Congrat, no dups detected.")
	}

	sort.Slice(dups, func(i, j int) bool {
		return dups[i].Similarity < dups[j].Similarity
	})

	highSimilarityCount := 0
	for _, dup := range dups {
		if dup.Similarity > similarityThreshold {
			highSimilarityCount++
		}
		if optimizeVerbosity > 2 {
			fmt.Printf("Duplicate detected for class %s in files (similarity: %.2f):\n", dup.ClassName, dup.Similarity)
			for _, path := range dup.Paths {
				fmt.Printf("  - %s\n", path)
			}
		}
	}
	log.Printf("duplicate classes detected          : %v", len(dups))
	log.Printf("duplicate with similarity over %.2f : %v", similarityThreshold, highSimilarityCount)
}

func init() {
	duplicateCmd.Flags().Float64VarP(&similarityThreshold, "similarity-threshold", "t", 0.6, "Threshold for similarity to count high similarity dup")
	duplicateCmd.Flags().IntVarP(&optimizeVerbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
	manifest.RequiredManifestFileFlag(duplicateCmd)
}
