package insight

import (
	"fmt"
	"log"
	"regexp"
	"sort"

	"federate/pkg/java"
	"federate/pkg/tabular"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var (
	classCamelCaseRegex = regexp.MustCompile(`[A-Z][a-z]+`)

	minCount    int
	useBarChart bool
	suffix      bool
	topK        int
)

type archetypeCount struct {
	name  string
	count int
}

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
	var idx = len(parts) - 1
	if !suffix {
		idx = 0
	}
	if len(parts) > 0 {
		return parts[idx]
	}
	return ""
}

func init() {
	taxonomyCmd.Flags().IntVarP(&minCount, "min-count", "m", 10, "Minimum count to display an element")
	taxonomyCmd.Flags().BoolVarP(&useBarChart, "bar-chart", "b", true, "Use bar chart display")
	taxonomyCmd.Flags().BoolVarP(&suffix, "suffix", "s", true, "Use suffix as archeType, else prefix")
	taxonomyCmd.Flags().IntVarP(&topK, "top", "t", 50, "Number of top elements to display in bar chart")
}

func printArchetypeAnalysis(taxonomy map[string]int) {
	var sorted []archetypeCount
	for name, count := range taxonomy {
		if count >= minCount {
			sorted = append(sorted, archetypeCount{name, count})
		}
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].count > sorted[j].count
	})

	fmt.Println("Code Taxonomy")
	fmt.Println("-------------")

	itemWidth := 25 // 每个项目的最大宽度
	width := util.TerminalWidth()
	itemsPerLine := width / itemWidth

	for i, cc := range sorted {
		if i > 0 && i%itemsPerLine == 0 {
			fmt.Println()
		}
		fmt.Printf("%-*s", itemWidth, fmt.Sprintf("%d %s", cc.count, cc.name))
	}

	if useBarChart {
		fmt.Println()
		printArchetypeBarChart(sorted)
	}
}

func printArchetypeBarChart(sorted []archetypeCount) {
	fmt.Println("Code Taxonomy (Top", topK, ")")
	fmt.Println("-----------------------------")

	items := make([]tabular.BarChartItem, len(sorted))
	for i, ac := range sorted {
		items[i] = tabular.BarChartItem{Name: ac.name, Count: ac.count}
	}

	tabular.PrintBarChart(items, topK, true)
}
