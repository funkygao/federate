package optimize

import (
	"log"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/optimizer"
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
	dups, err := optimizer.DetectDuplicateJava(m, similarityThreshold, similarityAlgo)
	if err != nil {
		log.Fatalf("Error detecting duplicate java: %v", err)
	}

	if len(dups) == 0 {
		log.Println("Congrat, no dups detected.")
	}

	sort.Slice(dups, func(i, j int) bool {
		return dups[i].Similarity < dups[j].Similarity
	})

	if optimizeVerbosity > 2 {
		for _, dup := range dups {
			log.Printf("%s %.2f", color.New(color.FgYellow).Sprintf(dup.ClassName), dup.Similarity)
			for _, path := range dup.Paths {
				log.Printf("  - %s\n", path)
			}
		}
	}
	log.Printf("Duplicate with similarity over %.2f : %v", similarityThreshold, len(dups))
}

func init() {
	manifest.RequiredManifestFileFlag(duplicateCmd)
	duplicateCmd.Flags().Float64VarP(&similarityThreshold, "similarity-threshold", "t", 0.72, "Threshold for similarity to count high similarity dup")
	duplicateCmd.Flags().IntVarP(&optimizeVerbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
	duplicateCmd.Flags().StringVarP(&similarityAlgo, "similarity-algorithm", "s", "simhash", "simhash | jaccard")
}
