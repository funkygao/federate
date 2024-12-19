package util

import (
	"encoding/json"
	"golang.org/x/term"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/width"
)

func Contains(s string, l []string) bool {
	for _, data := range l {
		if data == s {
			return true
		}
	}
	return false
}

func TerminalDisplayWidth(s string) int {
	width := 0
	for _, r := range s {
		width += runeWidth(r)
	}
	return width
}

func Truncate(s string, maxWidth int) string {
	if maxWidth <= 1 {
		return "…"
	}

	width := 0
	var truncated []rune

	for _, r := range s {
		charWidth := runeWidth(r)

		if width+charWidth > maxWidth {
			// 如果添加这个字符会超出最大宽度，停止并添加省略号
			return string(truncated) + "…"
		}

		width += charWidth
		truncated = append(truncated, r)
	}

	// 如果没有超出最大宽度，返回原始字符串
	return s
}

func runeWidth(r rune) int {
	if r == utf8.RuneError {
		return 1
	}

	switch width.LookupRune(r).Kind() {
	case width.EastAsianWide, width.EastAsianFullwidth:
		return 3 // 修改为 3，以匹配实际终端行为
	case width.EastAsianNarrow, width.EastAsianHalfwidth:
		return 1
	default:
		if r > unicode.MaxASCII || unicode.IsControl(r) {
			return 3 // 对于某些特殊字符，使用 3 个字符宽度
		}
		return 1
	}
}

func RemoveEmptyLines(s string) string {
	lines := strings.Split(s, "\n")
	var nonEmptyLines []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines = append(nonEmptyLines, line)
		}
	}
	return strings.Join(nonEmptyLines, "\n")
}

func CopyFile(sourceFile, targetFile string) error {
	source, err := os.Open(sourceFile)
	if err != nil {
		return err
	}
	defer source.Close()

	target, err := os.Create(targetFile)
	if err != nil {
		return err
	}
	defer target.Close()

	_, err = io.Copy(target, source)
	return err
}

func TerminalWidth() int {
	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		width = 80 // 默认宽度
	}
	return width
}

func GetSubdirectories(root string) ([]string, error) {
	var subdirs []string
	entries, err := ioutil.ReadDir(root)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			name := entry.Name()
			if len(name) > 2 && strings.HasPrefix(name, ".") {
				continue
			}
			subdirs = append(subdirs, filepath.Join(root, entry.Name()))
		}
	}

	return subdirs, nil
}

func FileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func DirExists(path string) bool {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return true
}

func IsDir(path string) bool {
	fileInfo, _ := os.Stat(path)
	return fileInfo.IsDir()
}

func UniqueStrings(input []string) []string {
	uniqueMap := make(map[string]bool)
	var result []string

	for _, str := range input {
		if _, exists := uniqueMap[str]; !exists {
			uniqueMap[str] = true
			result = append(result, str)
		}
	}

	return result
}

func Beautify(d any) string {
	b, _ := json.MarshalIndent(d, "", "  ")
	return string(b)
}
