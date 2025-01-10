package insight

import (
	"bufio"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"federate/pkg/apriori"
	"federate/pkg/tabular"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var (
	timeRange string
	topN      int
	verbosity int
	fileExt   string
)

type kv struct {
	Key   string
	Value int
}

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
	fmt.Println()

	removedCode := analyzeRemovedCode(dir, since)
	printSortedResults("Removed Code Analysis", removedCode, "File", "Times Removed")
	fmt.Println()

	fileAssociations := analyzeFileAssociations(dir, since)
	printFileAssociations("File Associations", fileAssociations)
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

func analyzeFileAssociations(dir, since string) apriori.Result {
	cmd := exec.Command("sh", "-c", fmt.Sprintf("git -C %s log %s --name-only --pretty=format:\"COMMIT%%n\"", dir, since))
	if verbosity > 1 {
		log.Printf("[%s] Executing: %s", dir, strings.Join(cmd.Args, " "))
	}
	output, err := cmd.Output()
	if err != nil {
		log.Printf("Error analyzing file associations: %v", err)
		return apriori.Result{}
	}

	var transactions []apriori.Transaction
	var currentTransaction apriori.Transaction
	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		if line == "COMMIT" {
			if len(currentTransaction) > 0 {
				transactions = append(transactions, currentTransaction)
				currentTransaction = apriori.Transaction{}
			}
		} else if line != "" && (fileExt == "" || filepath.Ext(line) == "."+fileExt) {
			currentTransaction = append(currentTransaction, line)
		}
	}
	if len(currentTransaction) > 0 {
		transactions = append(transactions, currentTransaction)
	}

	// 使用新的 Apriori 实现
	result := apriori.Run(transactions, 0.01, 0.5, true) // 最后的参数 true 表示使用 basename
	return result
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

	if fileExt == "java" {
		printTaxonomyStyle(title, sorted)
	}
}

func printFileAssociations(title string, result apriori.Result) {
	fmt.Println(title + ":")

	sort.Slice(result.Associations, func(i, j int) bool {
		return result.Associations[i].Confidence > result.Associations[j].Confidence
	})

	var cellData [][]string
	for i, assoc := range result.Associations {
		if i >= topN {
			break
		}
		cellData = append(cellData, []string{
			fmt.Sprintf("%s <-> %s", assoc.Items[0], assoc.Items[1]),
			fmt.Sprintf("%.2f%%", assoc.Confidence*100),
		})
	}

	tabular.Display([]string{"Association", "Confidence"}, cellData, false, -1)
	fmt.Println()
}

func printTaxonomyStyle(title string, sorted []kv) {
	title = "Taxonomy of " + title
	fmt.Println(title)
	fmt.Println(strings.Repeat("-", len(title)))

	width := util.TerminalWidth()

	// 使用 map 来聚合相同原型的文件
	archetypes := make(map[string]int)
	for _, item := range sorted[:min(len(sorted), topN)] {
		archetype := extractArchetype(item.Key)
		if archetype != "" {
			archetypes[archetype] += item.Value
		}
	}

	// 将聚合后的数据转换回 []kv 格式
	var archetypeSorted []kv
	for k, v := range archetypes {
		archetypeSorted = append(archetypeSorted, kv{k, v})
	}
	sort.Slice(archetypeSorted, func(i, j int) bool {
		return archetypeSorted[i].Value > archetypeSorted[j].Value
	})

	// 计算最长的项目长度
	maxLength := 0
	for _, item := range archetypeSorted {
		itemLength := utf8.RuneCountInString(fmt.Sprintf("%d %s", item.Value, item.Key))
		if itemLength > maxLength {
			maxLength = itemLength
		}
	}

	// 计算每行可以容纳的项目数和项目宽度
	itemWidth := maxLength + 2 // 添加一些额外的空间
	itemsPerLine := max(1, width/itemWidth)
	itemWidth = width / itemsPerLine // 重新调整以充分利用宽度

	for i, item := range archetypeSorted {
		if i >= topN {
			break
		}
		if i > 0 && i%itemsPerLine == 0 {
			fmt.Println()
		}
		fmt.Printf("%-*s", itemWidth, fmt.Sprintf("%d %s", item.Value, item.Key))
	}
	fmt.Println()
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
