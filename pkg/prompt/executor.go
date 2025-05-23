package prompt

import (
	"os"
	"strings"

	"federate/pkg/llm"
)

func executor(input string) {
	// 统一换行符，处理 Copy & Paste
	input = strings.ReplaceAll(input, "\r\n", "\n")
	input = strings.ReplaceAll(input, "\r", "\n")

	switch {
	case input == "go":
		promptGenerator.GenerateHighQualityPrompt()
		os.Exit(0)

	case input == "jd":
		r, _ := llm.CallRhino(promptGenerator.generatePrompt())
		println(r.Choices[0].Message.Content)
		os.Exit(0)

	case strings.HasPrefix(input, "!!"):
		promptGenerator.executeShellCommand(input[2:], true)
	case strings.HasPrefix(input, "!"):
		promptGenerator.executeShellCommand(input[1:], false)

	default:
		promptGenerator.AddInput(input)
		if Echo && promptGenerator.isMentionLine(input) {
			promptGenerator.echoMentions(input)
		}
	}
}
