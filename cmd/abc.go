package cmd

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"

	"federate/pkg/corpus"
	"federate/pkg/tablerender"
	"github.com/spf13/cobra"
)

var (
	maxLetters  int
	maxContexts int
	maxArticles int
	bzFile      string

	wikiMarkupRegex = regexp.MustCompile(`\{\{.*?\}\}|\[\[.*?\]\]|<.*?>`)
	urlRegex        = regexp.MustCompile(`https?://\S+`)
	nonLetterRegex  = regexp.MustCompile(`[^a-zA-Z\s.,!?]`)
)

var abcCmd = &cobra.Command{
	Use:   "abc",
	Short: "Analyze English letter distributions on a corpus",
	Long: `The abc command analyzes the distribution of letters preceding and following
the given letter combinations based on a corpus.

To use Simple English Wikipedia corpus, download from https://dumps.wikimedia.org/simplewiki/

Example usage:
  federate abc ee ea # use the default Brown Corpus
  federate abc ee ea -w ~/Downloads/simplewiki-20240901-pages-articles-multistream.xml.bz2`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			fmt.Println("Please provide at least one letter combination to analyze.")
			return
		}
		analyzeLetterDistributions(args)
	},
}

func analyzeLetterDistributions(combinations []string) {
	var (
		corpusText  string
		analyzeFunc func(text, combo string) (map[rune]int, map[rune]int, map[string]int)
	)

	if bzFile == "" {
		corpusText = corpus.GetCorpus()
		analyzeFunc = analyzeDistributionBrown
	} else {
		corpusText = corpus.GetSimplewikiCorpus(bzFile, maxArticles)
		analyzeFunc = analyzeDistributionWikipedia
	}

	var wg sync.WaitGroup
	rows := make([][]string, len(combinations))

	for i, combo := range combinations {
		wg.Add(1)
		go func(index int, letterCombo string) {
			defer wg.Done()
			preceding, following, contexts := analyzeFunc(corpusText, letterCombo)
			rows[index] = formatResult(letterCombo, preceding, following, contexts)
		}(i, combo)
	}

	wg.Wait()

	header := []string{"Combo", "Preceding", "Following", "Contexts"}
	tablerender.DisplayTable(header, rows, true, 0)
}

func analyzeDistributionWikipedia(text, combo string) (map[rune]int, map[rune]int, map[string]int) {
	preceding := make(map[rune]int)
	following := make(map[rune]int)
	contexts := make(map[string]int)

	sentences := strings.Split(text, ".")
	for _, sentence := range sentences {
		cleanSentence := cleanWikiText(sentence)
		if len(cleanSentence) < 30 || len(cleanSentence) > 150 {
			continue // 跳过过短或过长的句子
		}

		lowercaseSentence := strings.ToLower(cleanSentence)
		if strings.Contains(lowercaseSentence, combo) {
			// 检查是否包含至少3个不同的单词
			words := strings.Fields(lowercaseSentence)
			uniqueWords := make(map[string]bool)
			for _, word := range words {
				uniqueWords[word] = true
				if len(uniqueWords) >= 3 {
					break
				}
			}
			if len(uniqueWords) < 3 {
				continue
			}

			contexts[cleanSentence]++

			for _, word := range words {
				index := strings.Index(word, combo)
				if index != -1 {
					if index > 0 {
						letter := rune(word[index-1])
						if unicode.IsLetter(letter) {
							preceding[letter]++
						}
					}
					if index+len(combo) < len(word) {
						letter := rune(word[index+len(combo)])
						if unicode.IsLetter(letter) {
							following[letter]++
						}
					}
				}
			}
		}
	}

	return preceding, following, contexts
}

func analyzeDistributionBrown(text, combo string) (map[rune]int, map[rune]int, map[string]int) {
	preceding := make(map[rune]int)
	following := make(map[rune]int)
	contexts := make(map[string]int)

	words := strings.Fields(text)
	for _, word := range words {
		// 移除词性标注
		parts := strings.Split(word, "/")
		if len(parts) != 2 {
			continue
		}
		cleanWord := parts[0]
		pos := parts[1]

		index := strings.Index(cleanWord, combo)
		if index != -1 {
			if index > 0 {
				letter := rune(cleanWord[index-1])
				if unicode.IsLetter(letter) {
					preceding[letter]++
				}
			}
			if index+len(combo) < len(cleanWord) {
				letter := rune(cleanWord[index+len(combo)])
				if unicode.IsLetter(letter) {
					following[letter]++
				}
			}
			// 记录上下文
			context := fmt.Sprintf("%s (%s)", cleanWord, pos)
			contexts[context]++
		}
	}

	return preceding, following, contexts
}

func formatResult(combo string, preceding, following map[rune]int, contexts map[string]int) []string {
	if bzFile == "" {
		return []string{
			combo,
			formatDistribution(preceding),
			formatDistribution(following),
			formatContextsBrown(contexts),
		}
	} else {
		return []string{
			combo,
			formatDistribution(preceding),
			formatDistribution(following),
			formatContextsWikipedia(combo, contexts),
		}
	}
}

func formatDistribution(dist map[rune]int) string {
	type pair struct {
		letter rune
		count  int
	}

	pairs := make([]pair, 0, len(dist))
	for letter, count := range dist {
		pairs = append(pairs, pair{letter, count})
	}

	// Sort pairs by count in descending order
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].count > pairs[j].count
	})

	var result strings.Builder
	for i, p := range pairs {
		if i > 0 {
			result.WriteString(" ")
		}
		result.WriteString(fmt.Sprintf("%c/%d", p.letter, p.count))
		if i == maxLetters {
			break
		}
	}

	return result.String()
}

func formatContextsWikipedia(combo string, contexts map[string]int) string {
	type pair struct {
		context string
		count   int
	}

	pairs := make([]pair, 0, len(contexts))
	for context, count := range contexts {
		if strings.Count(context, " ") >= 5 { // 至少包含5个单词
			pairs = append(pairs, pair{context, count})
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].count > pairs[j].count
	})

	var result strings.Builder
	addedContexts := make(map[string]bool)
	for _, p := range pairs {
		if len(addedContexts) >= maxContexts {
			break
		}
		context := p.context
		if len(context) > 100 {
			words := strings.Fields(context)
			for j, word := range words {
				if strings.Contains(strings.ToLower(word), combo) {
					start := max(0, j-3)
					end := min(len(words), j+4)
					context = strings.Join(words[start:end], " ")
					if start > 0 {
						context = "... " + context
					}
					if end < len(words) {
						context = context + " ..."
					}
					break
				}
			}
		}
		// 检查是否已经添加了相似的上下文
		isDuplicate := false
		for addedContext := range addedContexts {
			if strings.Contains(addedContext, context) || strings.Contains(context, addedContext) {
				isDuplicate = true
				break
			}
		}
		if !isDuplicate {
			if len(addedContexts) > 0 {
				result.WriteString("\n")
			}
			result.WriteString(fmt.Sprintf("%s (%d)", context, p.count))
			addedContexts[context] = true
		}
	}

	if len(pairs) > len(addedContexts) {
		result.WriteString(fmt.Sprintf("\n... (%d more contexts)", len(pairs)-len(addedContexts)))
	}

	return result.String()
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func formatContextsBrown(contexts map[string]int) string {
	type pair struct {
		context string
		count   int
	}

	pairs := make([]pair, 0, len(contexts))
	for context, count := range contexts {
		pairs = append(pairs, pair{context, count})
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].count > pairs[j].count
	})

	var result strings.Builder
	for i, p := range pairs {
		if i > 0 {
			result.WriteString(" ")
		}
		parts := strings.Split(p.context, " ")
		if len(parts) == 2 {
			word := parts[0]
			result.WriteString(fmt.Sprintf("%s/%d", word, p.count))
		} else {
			result.WriteString(fmt.Sprintf("%s/%d", p.context, p.count))
		}
		if i == maxContexts-1 {
			break
		}
	}

	if len(pairs) > maxContexts {
		result.WriteString(fmt.Sprintf(" %d/...", len(pairs)-maxContexts))
	}

	return result.String()
}

func cleanWikiText(text string) string {
	// 移除维基标记
	text = wikiMarkupRegex.ReplaceAllString(text, "")
	// 移除URL
	text = urlRegex.ReplaceAllString(text, "")
	// 只保留字母、空格和基本标点
	text = nonLetterRegex.ReplaceAllString(text, " ")
	// 移除多余的空白字符
	text = strings.Join(strings.Fields(text), " ")
	// 移除重复单词
	words := strings.Fields(text)
	var result []string
	for i, word := range words {
		if i == 0 || strings.ToLower(word) != strings.ToLower(words[i-1]) {
			result = append(result, word)
		}
	}
	return text
}

func init() {
	abcCmd.Flags().IntVarP(&maxLetters, "max-letters", "m", 5, "How many context shown at most")
	abcCmd.Flags().IntVarP(&maxContexts, "max-contexts", "c", 5, "How many contexts shown at most")
	abcCmd.Flags().IntVarP(&maxArticles, "max-articles", "a", 5000, "How many articles to scan at most")
	abcCmd.Flags().StringVarP(&bzFile, "wikipedia", "w", "", "Downloaded simple wikipedia corpus bz2 file")
}
