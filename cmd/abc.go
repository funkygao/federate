package cmd

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"unicode"

	"federate/pkg/corpus"
	"federate/pkg/tablerender"
	"github.com/spf13/cobra"
)

var maxLetters int

var abcCmd = &cobra.Command{
	Use:   "abc",
	Short: "Analyze English letter distributions from a corpus",
	Long: `The abc command analyzes the distribution of letters preceding and following
the given letter combinations based on a corpus.

Example usage:
  federate abc ee ea`,
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			fmt.Println("Please provide at least one letter combination to analyze.")
			return
		}
		analyzeLetterDistributions(args)
	},
}

func analyzeLetterDistributions(combinations []string) {
	corpusText := corpus.GetCorpus()

	var wg sync.WaitGroup
	rows := make([][]string, len(combinations))

	for i, combo := range combinations {
		wg.Add(1)
		go func(index int, letterCombo string) {
			defer wg.Done()
			preceding, following := analyzeDistribution(corpusText, letterCombo)
			rows[index] = formatResult(letterCombo, preceding, following)
		}(i, combo)
	}

	wg.Wait()

	header := []string{"Combination", "Preceding Letters", "Following Letters"}
	tablerender.DisplayTable(header, rows, true, 0)
}

func analyzeDistribution(text, combo string) (map[rune]int, map[rune]int) {
	preceding := make(map[rune]int)
	following := make(map[rune]int)

	words := strings.Fields(text)
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

	return preceding, following
}

func formatResult(combo string, preceding, following map[rune]int) []string {
	return []string{
		combo,
		formatDistribution(preceding),
		formatDistribution(following),
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
		//result.WriteString(fmt.Sprintf("%c:%d", p.letter, p.count))
		result.WriteString(fmt.Sprintf("%c", p.letter))
		if i == maxLetters {
			break
		}
	}

	return result.String()
}

func init() {
	abcCmd.Flags().IntVarP(&maxLetters, "max-letters", "m", 5, "How many context shown at most")
}
