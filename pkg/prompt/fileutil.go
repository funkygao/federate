package prompt

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

var (
	ignorePatterns = []string{
		"*.log",
		"*.tmp",
		"*.jar",
		"*.war",
		"*.swp",
		".gitmodules",
		".gitignore",
		"*.ipynb",
		"go.mod",
		"go.sum",
		"*.swo",
		"*.iml",
		"*.ipr",
		"*.iws",
		"*.idea",
		"prompt.txt",
		".DS_Store",
	}

	ignoredDirs = []string{
		"target",
		"classes",
		".vscode",
		".git",
		".idea",
		".ipynb_checkpoints",
	}
)

// 检查文件是否是二进制文件
func isBinaryFile(content []byte) bool {
	for _, b := range content {
		r := rune(b)
		// 忽略 NUL 字符
		if r == 0x00 {
			continue
		}
		// 允许扩展 ASCII 字符
		if r >= 0x80 && r <= 0xFF {
			continue
		}
		if !unicode.IsPrint(r) && !unicode.IsSpace(r) {
			return true
		}
	}
	return false
}

func listFiles(dir string) ([]string, error) {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var fileList []string
	for _, file := range files {
		fileList = append(fileList, file.Name())
	}
	return fileList, nil
}

func readDirContent(dir string) (string, error) {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return "", err
	}

	var contentBuilder strings.Builder
	for _, file := range files {
		if !file.IsDir() {
			filePath := filepath.Join(dir, file.Name())
			content, err := ioutil.ReadFile(filePath)
			if err != nil {
				return "", err
			}
			if !isBinaryFile(content) {
				contentBuilder.WriteString(string(content))
				contentBuilder.WriteString("\n")
			}
		}
	}
	return contentBuilder.String(), nil
}

func readDirContentWithStructure(dir string, excludeList []string) (string, error) {
	var contentBuilder strings.Builder
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		fullPath, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		for _, excludePath := range excludeList {
			if fullPath == excludePath {
				if info.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}
		if info.IsDir() && shouldIgnoreDir(info.Name()) {
			return filepath.SkipDir
		}
		if !info.IsDir() {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			if isBinaryFile(content) || shouldIgnoreFile(path) {
				return nil
			}
			contentBuilder.WriteString(fmt.Sprintf("File: %s\nContent:\n```\n%s\n```\n\n", path, string(content)))
		} else {
			if path == "." {
				cwd, _ := os.Getwd()
				path = filepath.Base(cwd)
			}
			contentBuilder.WriteString(fmt.Sprintf("Dir: %s\n", path))
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	return contentBuilder.String(), nil
}

// 检查文件是否应该被忽略
func shouldIgnoreFile(filePath string) bool {
	for _, pattern := range ignorePatterns {
		matched, err := filepath.Match(pattern, filepath.Base(filePath))
		if err != nil {
			continue
		}
		if matched {
			return true
		}
	}
	return false
}

func shouldIgnoreDir(path string) bool {
	if strings.HasPrefix(path, ".") && len(path) > 2 {
		// hidden dir
		return true
	}

	for _, targetDir := range ignoredDirs {
		if strings.HasSuffix(path, targetDir) {
			return true
		}
	}
	return false
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

func recursiveSearch(root, pattern string, searchDir bool) ([]string, error) {
	var matches []string
	regexPattern := ".*" + strings.Join(strings.Split(pattern, ""), ".*") + ".*"
	re, err := regexp.Compile(regexPattern)
	if err != nil {
		return nil, err
	}

	err = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if shouldIgnoreDir(info.Name()) {
				return filepath.SkipDir
			}
			if searchDir && re.MatchString(info.Name()) {
				matches = append(matches, path)
			}
		} else if !searchDir && re.MatchString(info.Name()) && !shouldIgnoreFile(path) {
			matches = append(matches, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return matches, nil
}

func readFileContent(filePath string) (string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}
