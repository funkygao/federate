package util

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"federate/pkg/java"
	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var tfidfCmd = &cobra.Command{
	Use:   "tfidf [directory] [file extension]",
	Short: "Extract key features from specified files using TF-IDF",
	Long: `The tfidf command extracts key features from specified files in a Maven code repository using the TF-IDF algorithm.

Example usage:
  federate util tfidf ./my-maven-project`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) < 1 {
			fmt.Println("Please provide a directory.")
			return
		}
		directory := args[0]
		showKeyFeatures(directory)
	},
}

func showKeyFeatures(directory string) {
	files, err := getFiles(directory)
	if err != nil {
		fmt.Printf("Error getting files: %v\n", err)
		return
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

	// 使用表格形式展示结果
	header := []string{"File", "Top Features"}
	var rows [][]string

	for i, file := range files {
		topFeatures := getTopFeatures(tfidf[i], 5)
		row := []string{getFileNameWithoutExtension(file), topFeatures}
		rows = append(rows, row)
	}

	tablerender.DisplayTable(header, rows, false, 0)
}

func getFileNameWithoutExtension(filePath string) string {
	fileName := filepath.Base(filePath)
	return strings.TrimSuffix(fileName, filepath.Ext(fileName))
}

func getFiles(directory string) ([]string, error) {
	var files []string
	err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && java.IsJavaMainSource(info, path) {
			files = append(files, path)
		}
		return nil
	})
	return files, err
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
		topFeatures = append(topFeatures, util.Truncate(scores[i].word, 10))
	}

	return strings.Join(topFeatures, "  ")
}
