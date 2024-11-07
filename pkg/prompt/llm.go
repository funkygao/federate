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
		log.Printf("%s %v", path, err)
		return
	}

	fmt.Println(content)
}

func (pg *PromptGenerator) executeShellCommand(input string) {
	var cmd *exec.Cmd
	var outputToPrompt bool

	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/bash" // 默认使用 bash
	}

	if strings.HasPrefix(input, "!!") {
		cmd = exec.Command(shell, "-ic", input[2:])
		outputToPrompt = false
	} else {
		cmd = exec.Command(shell, "-ic", input[1:])
		outputToPrompt = true
	}

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	if err != nil {
		fmt.Printf("执行命令时出错: %v\n", err)
		return
	}

	output := out.String()
	fmt.Print(output)

	if outputToPrompt {
		promptGenerator.AddInput(fmt.Sprintf("Shell command: %s\nOutput:\n%s", input[1:], output))
	}
}

func (pg *PromptGenerator) processLineWithMention(line string) {
	matches := mentionRegex.FindAllStringSubmatch(line, -1)
	validLine := true

	for _, match := range matches {
		path := match[1]
		if isDir(path) {
			if !pg.processDirInput(path) {
				validLine = false
			}
		} else {
			if !pg.processFileInput(path) {
				validLine = false
			}
		}
	}

	if validLine {
		pg.buffer.WriteString(line)
		pg.buffer.WriteString("\n")
	}
}

func (pg *PromptGenerator) processDirInput(path string) bool {
	dir := strings.TrimSpace(path)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		log.Printf("skipped %v", err)
		return false
	}

	content, err := readDirContentWithStructure(dir)
	if err != nil {
		log.Printf("%s %v", dir, err)
		return false
	}

	pg.buffer.WriteString(content)
	return true
}

func (pg *PromptGenerator) processFileInput(path string) bool {
	file := strings.TrimSpace(path)
	if _, err := os.Stat(file); os.IsNotExist(err) {
		log.Printf("skipped %v", err)
		return false
	}

	content, err := readFileContent(file)
	if err != nil {
		log.Printf("%s %v", file, err)
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
	cmd := exec.Command("pbcopy")
	cmd.Stdin = strings.NewReader(finalContent)
	err := cmd.Run()
	if err != nil {
		log.Fatalf("%v", err)
	}

	tokenCount := CountTokensInK(finalContent)
	if Dump {
		fmt.Printf("生成的高质量 ChatGPT prompt 已保存到 %s，并已复制到剪贴板，约 %.2fK tokens\n", promptFile, tokenCount)
	} else {
		fmt.Printf("生成的高质量 ChatGPT prompt ，已复制到剪贴板，约 %.2fK tokens\n", tokenCount)
	}
}
