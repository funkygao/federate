package insight

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"

	"federate/pkg/java"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var (
	classCamelCaseRegex = regexp.MustCompile(`[A-Z][a-z]+`)
	minCount            int
)

var taxonomyCmd = &cobra.Command{
	Use:   "taxonomy [dir]",
	Short: "Create a comprehensive inventory of code elements",
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		analyzeTaxonomy(root)
	},
}

func analyzeTaxonomy(root string) {
	fileChan, errChan := java.ListJavaMainSourceFilesAsync(root)
	archetypes := make(map[string]int)
	for file := range fileChan {
		t := extractArchetype(file.Path)
		if t != "" {
			archetypes[t]++
		}
	}

	if err := <-errChan; err != nil {
		log.Fatalf("%v", err)
	}

	printArchetypeAnalysis(archetypes)
}

func extractArchetype(filename string) string {
	className := java.JavaFile2Class(filename)
	parts := classCamelCaseRegex.FindAllString(className, -1)
	if len(parts) > 0 {
		// 取最后1个
		return parts[len(parts)-1]
	}
	return ""
}

func printArchetypeAnalysis(taxonomy map[string]int) {
	type archetypeCount struct {
		name  string
		count int
	}

	var sorted []archetypeCount
	for name, count := range taxonomy {
		if count >= minCount {
			sorted = append(sorted, archetypeCount{name, count})
		}
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].count > sorted[j].count
	})

	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		width = 80 // 默认宽度
	}

	fmt.Println("Code Taxonomy")
	fmt.Println("-------------")

	itemWidth := 25 // 每个项目的最大宽度
	itemsPerLine := width / itemWidth

	for i, cc := range sorted {
		if i > 0 && i%itemsPerLine == 0 {
			fmt.Println()
		}
		fmt.Printf("%-*s", itemWidth, fmt.Sprintf("%d %s", cc.count, cc.name))
	}
	fmt.Println()
}

func init() {
	taxonomyCmd.Flags().IntVarP(&minCount, "min-count", "m", 10, "Minimum count to display an element")
}
