package prompt

import (
	"fmt"

	"github.com/c-bata/go-prompt"
)

var (
	Echo bool
	Dump bool

	baseDir         string
	promptGenerator *PromptGenerator
)

func Interact(codebaseDir string) {
	fmt.Printf("💡 输入 '%s' 引用文件，'%s' 引用目录，'go' 生成提示词，'go+' 增强提示词，'!cmd' 只执行不记录，'!!cmd' 执行命令，Ctrl+D 退出\n", filePrefix, dirPrefix)

	baseDir = codebaseDir
	promptGenerator = NewPromptGenerator()

	var prefixOption prompt.Option
	if Echo {
		prefixOption = prompt.OptionPrefix("> ")
	} else {
		prefixOption = prompt.OptionLivePrefix(livePrefix)
	}

	p := prompt.New(
		executor,
		completer,
		prompt.OptionTitle("ChatGPT Prompt 生成器"),
		prompt.OptionPrefixTextColor(prompt.Blue),
		prompt.OptionInputTextColor(prompt.Brown),
		prefixOption,
		prompt.OptionAddKeyBind(prompt.KeyBind{
			Key: prompt.ControlD,
			Fn: func(buf *prompt.Buffer) {
				buf.InsertText("\n", false, true)
			},
		}),
	)
	p.Run()
}

// 在每次输入时都会被调用，而不是在每次换行时
func livePrefix() (string, bool) {
	if Echo &&
		len(promptGenerator.lastLine) > 1 &&
		promptGenerator.isMentionLine(promptGenerator.lastLine) {
		prefix := "> " + promptGenerator.lastLine + "\n> "
		promptGenerator.ResetLastLine()
		return prefix, true
	}

	return "> ", true
}
