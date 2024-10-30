package util

import (
	"io"
	"os"
	"strings"
)

func Contains(s string, l []string) bool {
	for _, data := range l {
		if data == s {
			return true
		}
	}
	return false
}

func Truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
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
