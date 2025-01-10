package insight

import (
	"fmt"
	"io/ioutil"
	"math"
	"regexp"
	"sort"
	"strings"

	"federate/pkg/java"
	"federate/pkg/tabular"
	"github.com/spf13/cobra"
)

var featureN int

var tfidfCmd = &cobra.Command{
	Use:   "tfidf <dir>",
	Short: "Extract key features from specified files using TF-IDF",
	Long: `The tfidf command extracts key features from specified files in a Maven code repository using the TF-IDF algorithm.

Example usage:
  federate util tfidf ./my-maven-project`,
	Run: func(cmd *cobra.Command, args []string) {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}
		showKeyFeatures(root)
	},
}

func showKeyFeatures(directory string) {
	fileChan, _ := java.ListJavaMainSourceFilesAsync(directory)
	var files []string
	for f := range fileChan {
		files = append(files, f.Path)
	}

	corpus := make([]string, len(files))
	for i, file := range files {
		content, err := ioutil.ReadFile(file)
		if err != nil {
			fmt.Printf("Error reading file %s: %v\n", file, err)
			continue
		}
		corpus[i] = string(content)
	}

	tfidf := calculateTFIDF(corpus)

	var data [][]string
	header := []string{"Class", "Features"}
	for i, file := range files {
		topFeatures := getTopFeatures(tfidf[i], featureN)
		data = append(data, []string{java.JavaFile2Class(file), topFeatures})
	}
	tabular.Display(header, data, true, 0)
}

func calculateTFIDF(corpus []string) []map[string]float64 {
	documentCount := len(corpus)
	wordFrequency := make([]map[string]int, documentCount)
	documentFrequency := make(map[string]int)

	for i, document := range corpus {
		words := tokenize(document)
		wordFrequency[i] = make(map[string]int)
		for _, word := range words {
			wordFrequency[i][word]++
			if wordFrequency[i][word] == 1 {
				documentFrequency[word]++
			}
		}
	}

	tfidf := make([]map[string]float64, documentCount)
	for i := range tfidf {
		tfidf[i] = make(map[string]float64)
	}

	for i, document := range corpus {
		words := tokenize(document)
		for _, word := range words {
			tf := float64(wordFrequency[i][word]) / float64(len(words))
			idf := math.Log(float64(documentCount) / float64(documentFrequency[word]))
			tfidf[i][word] = tf * idf
		}
	}

	return tfidf
}

func tokenize(text string) []string {
	// 移除注释
	text = removeComments(text)

	// 分割驼峰命名
	text = splitCamelCase(text)

	// 分割单词
	words := strings.Fields(text)

	// 移除标点符号和转换为小写
	var result []string
	for _, word := range words {
		word = strings.ToLower(word)
		word = strings.Trim(word, ".,;:!?()[]{}\"'")
		if word != "" {
			result = append(result, word)
		}
	}

	return result
}

func removeComments(text string) string {
	// 移除单行注释
	singleLineComment := regexp.MustCompile(`//.*`)
	text = singleLineComment.ReplaceAllString(text, "")

	// 移除多行注释
	multiLineComment := regexp.MustCompile(`/\*[\s\S]*?\*/`)
	text = multiLineComment.ReplaceAllString(text, "")

	return text
}

func splitCamelCase(text string) string {
	var words []string
	lastIndex := 0
	for i := 0; i < len(text); i++ {
		if i > 0 && isUpperCase(text[i]) && !isUpperCase(text[i-1]) {
			words = append(words, text[lastIndex:i])
			lastIndex = i
		}
	}
	words = append(words, text[lastIndex:])
	return strings.Join(words, " ")
}

func isUpperCase(c byte) bool {
	return c >= 'A' && c <= 'Z'
}

func getTopFeatures(tfidf map[string]float64, n int) string {
	type wordScore struct {
		word  string
		score float64
	}

	var scores []wordScore
	for word, score := range tfidf {
		scores = append(scores, wordScore{word, score})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	var topFeatures []string
	for i := 0; i < n && i < len(scores); i++ {
		topFeatures = append(topFeatures, strings.ReplaceAll(scores[i].word, "\n", ""))
	}

	return strings.Join(topFeatures, "  ")
}

func init() {
	tfidfCmd.Flags().IntVarP(&featureN, "features", "n", 10, "How many features to extract")
}
