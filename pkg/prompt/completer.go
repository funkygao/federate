package prompt

import (
	"strings"

	"github.com/c-bata/go-prompt"
)

const (
	atPrefix   = "@"
	filePrefix = atPrefix + "f"
	dirPrefix  = atPrefix + "d"
	rulePrefix = atPrefix + "r"
	killPrefix = atPrefix + "k"
	targetFile = "prompt.txt"
)

func completer(d prompt.Document) []prompt.Suggest {
	text := d.TextBeforeCursor()
	switch {
	case strings.Contains(text, filePrefix):
		return handleFileCompletion(text)
	case strings.Contains(text, dirPrefix):
		return handleDirCompletion(text)
	case strings.HasPrefix(text, rulePrefix):
		return handleRuleCompletion(text)
	case strings.HasPrefix(text, killPrefix):
		return handleKillCompletion(text)
	default:
		return []prompt.Suggest{}
	}
}

func handleRuleCompletion(text string) []prompt.Suggest {
	suggestions := []prompt.Suggest{}
	for _, rule := range rules {
		if strings.HasPrefix(rule.Name, strings.TrimPrefix(text, rulePrefix)) {
			suggestions = append(suggestions, prompt.Suggest{Text: rulePrefix + rule.Name})
		}
	}
	return suggestions
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

func handleKillCompletion(text string) []prompt.Suggest {
	prefix := text[strings.LastIndex(text, killPrefix):]
	matches, _ := recursiveSearch(baseDir, strings.TrimPrefix(prefix, killPrefix), false)

	suggestions := []prompt.Suggest{}
	for _, match := range matches {
		suggestions = append(suggestions, prompt.Suggest{Text: atPrefix + "k" + match})
	}
	return suggestions
}
