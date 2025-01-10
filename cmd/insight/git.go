package insight

import (
	"bufio"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"github.com/spf13/cobra"
)

var (
	timeRange string
	topN      int
	verbosity int
	fileExt   string
)

var gitInsightCmd = &cobra.Command{
	Use:   "git [dir]",
	Short: "Analyze Git history for valuable insights",
	Run: func(cmd *cobra.Command, args []string) {
		dir := "."
		if len(args) > 0 {
			dir = args[0]
		}
		analyzeGitHistory(dir)
	},
}

func analyzeGitHistory(dir string) {
	if !isGitRepo(dir) {
		log.Fatalf("%s is not a git repository", dir)
	}

	since := getSinceDate(timeRange)

	hotspots := analyzeChangeFrequency(dir, since)
	printSortedResults("Code Hotspots (Most frequently changed files)", hotspots, "File", "Commits")

	removedCode := analyzeRemovedCode(dir, since)
	printSortedResults("Removed Code Analysis", removedCode, "File", "Times Removed")
}

func analyzeChangeFrequency(dir, since string) map[string]int {
	cmd := exec.Command("sh", "-c", fmt.Sprintf("git -C %s log %s --name-only --pretty=format:", dir, since))
	if verbosity > 1 {
		log.Printf("[%s] Executing: %s", dir, strings.Join(cmd.Args, " "))
	}
	output, err := cmd.Output()
	if err != nil {
		log.Printf("Error analyzing change frequency: %v", err)
		return nil
	}

	files := make(map[string]int)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		file := scanner.Text()
		if file != "" && (fileExt == "" || filepath.Ext(file) == "."+fileExt) {
			files[file]++
		}
	}

	return files
}

func analyzeRemovedCode(dir, since string) map[string]int {
	cmd := exec.Command("sh", "-c", fmt.Sprintf("git -C %s log %s --diff-filter=D --summary", dir, since))
	if verbosity > 1 {
		log.Printf("[%s] Executing: %s", dir, strings.Join(cmd.Args, " "))
	}
	output, err := cmd.Output()
	if err != nil {
		log.Printf("Error analyzing removed code: %v", err)
		return nil
	}

	removedFiles := make(map[string]int)
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "delete mode") {
			parts := strings.Fields(line)
			if len(parts) > 3 {
				file := parts[3]
				if fileExt == "" || filepath.Ext(file) == "."+fileExt {
					removedFiles[file]++
				}
			}
		}
	}

	return removedFiles
}

func printSortedResults(title string, data map[string]int, keyHeader, valueHeader string) {
	type kv struct {
		Key   string
		Value int
	}

	var sorted []kv
	for k, v := range data {
		sorted = append(sorted, kv{k, v})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Value > sorted[j].Value
	})

	var cellData [][]string
	for i, kv := range sorted {
		if i >= topN {
			break
		}
		cellData = append(cellData, []string{kv.Key, fmt.Sprintf("%d", kv.Value)})
	}

	log.Println(title + ":")
	tabular.Display([]string{keyHeader, valueHeader}, cellData, false, -1)
}

func isGitRepo(dir string) bool {
	cmd := exec.Command("git", "-C", dir, "rev-parse", "--is-inside-work-tree")
	err := cmd.Run()
	return err == nil
}

func getSinceDate(timeRange string) string {
	switch timeRange {
	case "1m":
		return "--since='1 month ago'"
	case "1y":
		return "--since='1 year ago'"
	case "all":
		return ""
	default:
		log.Fatalf("Invalid time range: %s", timeRange)
		return ""
	}
}

func init() {
	gitInsightCmd.Flags().StringVarP(&timeRange, "time-range", "t", "all", "Time range for analysis: 1m (1 month), 1y (1 year), all (all history)")
	gitInsightCmd.Flags().IntVarP(&topN, "top", "n", 20, "Number of top results to display")
	gitInsightCmd.Flags().IntVarP(&verbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
	gitInsightCmd.Flags().StringVarP(&fileExt, "ext", "e", "", "File extension to filter (e.g., 'java' for .java files)")
}
