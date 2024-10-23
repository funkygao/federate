package prompt

import (
	"strings"

	"github.com/c-bata/go-prompt"
)

const (
	atPrefix   = "@"
	filePrefix = atPrefix + "f"
	dirPrefix  = atPrefix + "d"
	targetFile = "prompt.txt"
)

func completer(d prompt.Document) []prompt.Suggest {
	text := d.TextBeforeCursor()
	switch {
	case strings.Contains(text, filePrefix):
		return handleFileCompletion(text)
	case strings.Contains(text, dirPrefix):
		return handleDirCompletion(text)
	default:
		return []prompt.Suggest{}
	}
}

func handleFileCompletion(text string) []prompt.Suggest {
	prefix := text[strings.LastIndex(text, filePrefix):]
	matches, _ := recursiveSearch(baseDir, strings.TrimPrefix(prefix, filePrefix), false)

	suggestions := []prompt.Suggest{}
	for _, match := range matches {
		suggestions = append(suggestions, prompt.Suggest{Text: atPrefix + match})
	}
	return suggestions
}

func handleDirCompletion(text string) []prompt.Suggest {
	prefix := text[strings.LastIndex(text, dirPrefix):]
	matches, _ := recursiveSearch(baseDir, strings.TrimPrefix(prefix, dirPrefix), true)

	suggestions := []prompt.Suggest{}
	for _, match := range matches {
		if isDir(match) {
			suggestions = append(suggestions, prompt.Suggest{Text: atPrefix + match})
		}
	}
	return suggestions
}
