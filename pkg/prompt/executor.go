package prompt

import (
	"strings"
)

func executor(input string) {
	// 统一换行符，处理 Copy & Paste
	input = strings.ReplaceAll(input, "\r\n", "\n")
	input = strings.ReplaceAll(input, "\r", "\n")

	switch {
	case input == "go":
		promptGenerator.GenerateHighQualityPrompt(true)
	case input == "gon":
		promptGenerator.GenerateHighQualityPrompt(false)
	case strings.HasPrefix(input, "!"):
		promptGenerator.executeShellCommand(input)
	default:
		promptGenerator.AddInput(input)
		if Echo && promptGenerator.isMentionLine(input) {
			promptGenerator.echoMentions(input)
		}
	}
}
