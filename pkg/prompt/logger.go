package prompt

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"

	"federate/pkg/util"
)

type PromptLogger struct {
	buffer bytes.Buffer
}

func NewPromptLogger() *PromptLogger {
	return &PromptLogger{}
}

func (pl *PromptLogger) Start() {
	log.SetOutput(io.MultiWriter(os.Stdout, &pl.buffer))
}

func (pl *PromptLogger) Stop() {
	log.SetOutput(os.Stdout)

	promptContent := pl.buffer.String()
	if err := util.ClipboardPut(promptContent); err == nil {
		log.Printf("ChatGPT Prompt 已复制到剪贴板，约 %.2fK tokens", CountTokensInK(promptContent))
	} else {
		log.Printf("复制到剪贴板失败: %v", err)
	}
}

func (pl *PromptLogger) AddPrompt(format string, v ...interface{}) {
	fmt.Fprintf(&pl.buffer, format, v...)
}
