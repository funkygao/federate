package prompt

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/util"
)

const (
	promptFile = "prompt.txt"
)

var mentionRegex = regexp.MustCompile(`@(\S+)`)

type PromptGenerator struct {
	lastLine    string
	buffer      strings.Builder
	rule        *Rule
	excludeList []string
}

func NewPromptGenerator() *PromptGenerator {
	return &PromptGenerator{
		excludeList: []string{},
	}
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
		if strings.HasPrefix(path, "r") {
			validMention = pg.processRuleInput(path[1:])
		} else if strings.HasPrefix(path, "k") {
			validMention = pg.processKillInput(path[1:])
		} else if isDir(path) {
			validMention = pg.processDirInput(path)
		} else {
			validMention = pg.processFileInput(path)
		}
	}

	if !validMention {
		pg.buffer.WriteString(line)
		pg.buffer.WriteString("\n")
	}
}

func (pg *PromptGenerator) processKillInput(path string) bool {
	fullPath, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	pg.excludeList = append(pg.excludeList, fullPath)
	return true
}

func (pg *PromptGenerator) processRuleInput(ruleName string) bool {
	for _, rule := range rules {
		if rule.Name == ruleName {
			pg.rule = &rule
			return true
		}
	}
	return false
}

func (pg *PromptGenerator) processDirInput(path string) bool {
	dir := strings.TrimSpace(path)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return false
	}

	content, err := readDirContentWithStructure(dir, pg.excludeList)
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

func (pg *PromptGenerator) GenerateHighQualityPrompt() {
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

	finalContent := pg.buffer.String()
	if pg.rule != nil {
		finalContent = pg.rule.SystemPrompt + "\n\n" + pg.rule.UserPrompt + "\n\n" + finalContent
	}

	if Dump {
		if err := ioutil.WriteFile(promptFile, []byte(finalContent), 0644); err != nil {
			log.Fatalf("%v", err)
		}
	}

	// 将内容复制到剪贴板
	if err := util.ClipboardPut(finalContent); err != nil {
		log.Fatalf("%v", err)
	}

	tokenCount := CountTokensInK(finalContent)
	if Dump {
		fmt.Printf("ChatGPT Prompt 已保存到 %s，并已复制到剪贴板，约 %.2fK tokens\n", promptFile, tokenCount)
	} else {
		fmt.Printf("ChatGPT Prompt 已复制到剪贴板，约 %.2fK tokens\n", tokenCount)
	}
}
