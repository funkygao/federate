package microservice

import (
	"fmt"
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/scanner"
	"github.com/spf13/cobra"
)

var similarityThreshold float64

var optimizeCmd = &cobra.Command{
	Use:   "optimize",
	Short: "Identify potential areas for optimization from components",
	Long: `The optimize command identify potential areas for optimization from components.
It detects duplicated java classes with similarity score.

Example usage:
  federate microservice optimize -i manifest.yaml --similarity-threshold 0.5`,
	Run: func(cmd *cobra.Command, args []string) {
		manifest, err := manifest.LoadManifest()
		if err != nil {
			log.Fatalf("Error loading manifest: %v", err)
		}

		optimize(manifest)
	},
}

func optimize(m *manifest.Manifest) {
	showDuplicates(m)
}

func showDuplicates(m *manifest.Manifest) {
	classMap, err := scanner.ScanComponents(m)
	if err != nil {
		log.Fatalf("Error scanning components: %v", err)
	}
	dups, err := scanner.CheckDup(classMap)
	if err != nil {
		sort.Slice(dups, func(i, j int) bool {
			return dups[i].Similarity < dups[j].Similarity
		})

		highSimilarityCount := 0
		for _, dup := range dups {
			if dup.Similarity > similarityThreshold {
				highSimilarityCount++
			}
			fmt.Printf("Duplicate detected for class %s in files (similarity: %.2f):\n", dup.ClassName, dup.Similarity)
			for _, path := range dup.Paths {
				fmt.Printf("  - %s\n", path)
			}
		}
		log.Printf("duplicate classes detected          : %v", len(dups))
		log.Printf("duplicate with similarity over %.2f : %v", similarityThreshold, highSimilarityCount)
	} else {
		log.Println("No dups detected.")
	}
}

func init() {
	optimizeCmd.Flags().Float64VarP(&similarityThreshold, "similarity-threshold", "t", 0.6, "Threshold for similarity to count high similarity dup")
	manifest.RequiredManifestFileFlag(optimizeCmd)
}
