package debug

import (
	"fmt"
	"os"
	"strings"

	"federate/pkg/java"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

var wordRegexp bool

var ymlCmd = &cobra.Command{
	Use:   "yml <key> [dir]",
	Short: "Search for a specified key in YAML files within a directory",
	Long: `The yml command searches for a specified key in YAML files within a specified directory recursively.

Example usage:
  federate debug yml spring.profiles.active [dir]`,
	Args: cobra.RangeArgs(1, 2),
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) < 1 {
			fmt.Println("Please provide a key to search for.")
			return
		}
		key := args[0]
		dir := "."
		if len(args) > 1 {
			dir = args[1]
		}
		searchYAMLFiles(dir, key)
	},
}

func searchYAMLFiles(dir, key string) {
	fileChan, _ := java.ListFilesAsync(dir, func(info os.FileInfo, path string) bool {
		return strings.HasSuffix(info.Name(), ".yaml") || strings.HasSuffix(info.Name(), ".yml")
	})
	for f := range fileChan {
		searchYAMLFile(f.Path, key)
	}
}

func searchYAMLFile(filePath, key string) {
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("Error opening file %s: %v\n", filePath, err)
		return
	}
	defer file.Close()

	var data map[string]any
	decoder := yaml.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		if !strings.Contains(err.Error(), "found character that cannot start any token") && err.Error() != "EOF" {
			fmt.Printf("%s: %v\n", filePath, err)
		}
		return
	}

	if results := findPartialKey(data, key, ""); len(results) > 0 {
		for _, result := range results {
			coloredKey := color.New(color.FgRed).SprintFunc()(key)
			highlightedKey := strings.Replace(result.key, key, coloredKey, -1)
			fmt.Printf("%s: %s = %v\n", filePath, highlightedKey, result.value)
		}
	}
}

type searchResult struct {
	key   string
	value any
}

func findPartialKey(data map[string]any, key, prefix string) []searchResult {
	var results []searchResult
	for k, v := range data {
		fullKey := k
		if prefix != "" {
			fullKey = prefix + "." + k
		}

		if shouldMatch(fullKey, key) {
			results = append(results, searchResult{key: fullKey, value: v})
		}
		if nestedMap, ok := v.(map[any]any); ok {
			results = append(results, findPartialKey(convertMap(nestedMap), key, fullKey)...)
		} else if nestedMap, ok := v.(map[string]any); ok {
			results = append(results, findPartialKey(nestedMap, key, fullKey)...)
		}
	}
	return results
}

func shouldMatch(fullKey, key string) bool {
	if wordRegexp {
		return matchWholeWord(fullKey, key)
	}
	return strings.Contains(fullKey, key)
}

func matchWholeWord(fullKey, key string) bool {
	parts := strings.Split(fullKey, ".")
	for _, part := range parts {
		if part == key {
			return true
		}
	}
	return false
}

func convertMap(input map[any]any) map[string]any {
	output := make(map[string]any)
	for k, v := range input {
		output[fmt.Sprintf("%v", k)] = v
	}
	return output
}

func init() {
	ymlCmd.Flags().BoolVarP(&wordRegexp, "word-regexp", "w", false, "Only match whole words")
}
