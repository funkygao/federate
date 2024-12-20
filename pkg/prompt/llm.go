package prompt

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/atotto/clipboard"
)

const (
	systemPrompt = "You are an excellent programming expert. Your task is to provide high-quality code modification suggestions or explain technical principles in detail."

	userBasePrompt = `Instructions:
1. Carefully analyze the original code.
2. When suggesting modifications:
   - Explain why the change is necessary or beneficial.
   - Identify and explain the root cause of the issue, not just add new solutions on top of existing ones.
3. Guidelines for suggested code snippets:
   - Only apply the change(s) suggested by the most recent assistant message.
   - Do not make any unrelated changes to the code.
   - Produce a valid full rewrite of the entire original file without skipping any lines. Do not be lazy!
   - Do not omit large parts of the original file without reason.
   - Do not omit any needed changes from the requisite messages/code blocks.
   - Keep your suggested code changes minimal, and do not include irrelevant lines.
   - Review all suggestions, ensuring your modification is high quality.
4. Ensure the generated code adheres to:
   - Object-oriented principles
   - SOLID principles
   - Simplicity
   - Extensibility
   - Maintainability
   - Readability
   - Code style consistency (naming conventions, comments, etc.)
   - Performance optimization
5. Only provide the code that needs to be modified, do not include unchanged code.
`

	promptFile = "prompt.txt"
)

var mentionRegex = regexp.MustCompile(`@(\S+)`)

type PromptGenerator struct {
	lastLine string
	buffer   strings.Builder
}

func NewPromptGenerator() *PromptGenerator {
	return &PromptGenerator{}
}

func (pg *PromptGenerator) ResetLastLine() {
	pg.lastLine = ""
}

func (pg *PromptGenerator) AddInput(input string) {
	pg.buffer.WriteString(input)
	pg.buffer.WriteString("\n")
	pg.lastLine = input
}

func (pg *PromptGenerator) isMentionLine(line string) bool {
	return mentionRegex.MatchString(line)
}

func (pg *PromptGenerator) echoMentions(line string) {
	matches := mentionRegex.FindAllStringSubmatch(line, -1)
	echo := false
	for _, match := range matches {
		path := match[1]
		if !isDir(path) {
			pg.echoFileInput(path)
			echo = true
		}
	}
	if !echo {
		pg.ResetLastLine()
	}
}

func (pg *PromptGenerator) echoFileInput(path string) {
	content, err := readFileContent(strings.TrimSpace(path))
	if err != nil {
		return
	}

	fmt.Println(content)
}

func (pg *PromptGenerator) executeShellCommand(command string, outputToPrompt bool) {
	// 将命令拆分为程序名和参数
	parts := strings.Fields(command)
	if len(parts) == 0 {
		fmt.Println("错误：空命令")
		return
	}

	cmd := exec.Command(parts[0], parts[1:]...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	cmd.Run()

	stdoutStr := stdout.String()
	stderrStr := stderr.String()

	if stdoutStr != "" {
		fmt.Print(stdoutStr)
	}

	if stderrStr != "" {
		fmt.Print(stderrStr)
	}

	if outputToPrompt {
		promptContent := fmt.Sprintf("Shell command: %s\n", command)
		if stdoutStr != "" {
			promptContent += fmt.Sprintf("Standard output:\n```\n%s```\n", stdoutStr)
		}
		if stderrStr != "" {
			promptContent += fmt.Sprintf("Standard error:\n```\n%s```\n", stderrStr)
		}

		promptGenerator.AddInput(promptContent)
	}
}

func (pg *PromptGenerator) processLineWithMention(line string) {
	matches := mentionRegex.FindAllStringSubmatch(line, -1)

	validMention := false
	for _, match := range matches {
		path := match[1]
		if isDir(path) {
			validMention = pg.processDirInput(path)
		} else {
			validMention = pg.processFileInput(path)
		}
	}

	if !validMention {
		// 用户 copy & paste 了包含 @ 的内容，而这部分内容并不希望 mention file/dir
		pg.buffer.WriteString(line)
		pg.buffer.WriteString("\n")
	}
}

func (pg *PromptGenerator) processDirInput(path string) bool {
	dir := strings.TrimSpace(path)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return false
	}

	content, err := readDirContentWithStructure(dir)
	if err != nil {
		return false
	}

	pg.buffer.WriteString(content)
	return true
}

func (pg *PromptGenerator) processFileInput(path string) bool {
	file := strings.TrimSpace(path)
	if _, err := os.Stat(file); os.IsNotExist(err) {
		return false
	}

	content, err := readFileContent(file)
	if err != nil {
		return false
	}
	pg.buffer.WriteString(fmt.Sprintf("File: %s\nContent:\n```\n%s\n```\n\n", file, content))
	return true
}

func (pg *PromptGenerator) GenerateHighQualityPrompt(useTemplate bool) {
	// 先写入 systemPrompt 和 userBasePrompt
	initialBuffer := strings.Builder{}
	if useTemplate {
		initialBuffer.WriteString(systemPrompt)
		initialBuffer.WriteString("\n\n")
		initialBuffer.WriteString(userBasePrompt)
		initialBuffer.WriteString("\n\n")
	}

	lines := strings.Split(pg.buffer.String(), "\n")
	pg.buffer.Reset()

	for _, line := range lines {
		if pg.isMentionLine(line) {
			pg.processLineWithMention(line)
		} else {
			pg.buffer.WriteString(line)
			pg.buffer.WriteString("\n")
		}
	}

	if pg.buffer.Len() == 0 {
		fmt.Println("没有内容可保存。")
		return
	}

	// 将 initialBuffer 的内容添加到 pg.buffer 的最前面
	finalContent := initialBuffer.String() + pg.buffer.String()

	if Dump {
		if err := ioutil.WriteFile(promptFile, []byte(finalContent), 0644); err != nil {
			log.Fatalf("%v", err)
		}
	}

	// 将内容复制到剪贴板
	if err := clipboard.WriteAll(finalContent); err != nil {
		log.Fatalf("%v", err)
	}

	tokenCount := CountTokensInK(finalContent)
	if Dump {
		fmt.Printf("ChatGPT Prompt 已保存到 %s，并已复制到剪贴板，约 %.2fK tokens\n", promptFile, tokenCount)
	} else {
		fmt.Printf("ChatGPT Prompt 已复制到剪贴板，约 %.2fK tokens\n", tokenCount)
	}
}
