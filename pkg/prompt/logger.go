package prompt

import (
	"bytes"
	"io"
	"log"
	"os"
)

type PromptLogger struct {
	buffer bytes.Buffer
}

func NewPromptLogger() *PromptLogger {
	logger := &PromptLogger{}
	log.SetOutput(io.MultiWriter(os.Stdout, &logger.buffer))
	return logger
}

// Append to prompt only.
func (p *PromptGenerator) AppendPrompt() {
}

// Append both to log and prompt.
func (p *PromptGenerator) Append() {
}
