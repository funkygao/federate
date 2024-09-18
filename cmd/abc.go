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
	maxLetters      int
	maxContexts     int
	bzFile          string
	wikiMarkupRegex = regexp.MustCompile(`\{\{.*?\}\}|\[\[.*?\]\]|<.*?>`)
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
		corpusText = corpus.GetSimplewikiCorpus(bzFile)
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
		if len(cleanSentence) < 10 || len(cleanSentence) > 200 {
			continue // 跳过过短或过长的句子
		}

		words := strings.Fields(cleanSentence)
		for _, word := range words {
			cleanWord := strings.ToLower(word)
			cleanWord = strings.Trim(cleanWord, ".,!?:;\"'()[]{}") // 移除标点符号

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
				// 记录整个句子作为上下文
				contexts[cleanSentence]++
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
		if strings.Contains(context, combo) && len(context) > 20 {
			pairs = append(pairs, pair{context, count})
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].count > pairs[j].count
	})

	var result strings.Builder
	for i, p := range pairs {
		if i > 0 {
			result.WriteString("\n")
		}
		context := p.context
		if len(context) > 100 {
			context = context[:97] + "..."
		}
		result.WriteString(fmt.Sprintf("%s (%d)", context, p.count))
		if i == maxContexts-1 {
			break
		}
	}

	if len(pairs) > maxContexts {
		result.WriteString(fmt.Sprintf("\n... (%d more contexts)", len(pairs)-maxContexts))
	}

	return result.String()
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
	// 移除多余的空白字符
	text = strings.Join(strings.Fields(text), " ")
	return text
}

func init() {
	abcCmd.Flags().IntVarP(&maxLetters, "max-letters", "m", 5, "How many context shown at most")
	abcCmd.Flags().IntVarP(&maxContexts, "max-contexts", "c", 5, "How many contexts shown at most")
	abcCmd.Flags().StringVarP(&bzFile, "wikipedia", "w", "", "Downloaded simple wikipedia corpus bz2 file")
}
