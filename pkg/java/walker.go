package java

import (
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/primitive"
)

type FileInfo struct {
	Path string
	Info os.FileInfo
}

func ListXMLFilesAsync(root string) (<-chan FileInfo, <-chan error) {
	return ListFilesAsync(root, IsXML)
}

func ListResourceFilesAsync(root string) (<-chan FileInfo, <-chan error) {
	return ListFilesAsync(root, IsResourceFile)
}

func ListJavaMainSourceFilesAsync(root string) (<-chan FileInfo, <-chan error) {
	return ListFilesAsync(root, IsJavaMainSource)
}

func ListFilesAsync(root string, predicate func(info os.FileInfo, path string) bool) (<-chan FileInfo, <-chan error) {
	return ListFilesAsync_(root, predicate, walkDir)
}

func ListFilesAsync_(root string, predicate func(info os.FileInfo, path string) bool, dirWalk func(info os.FileInfo) error) (<-chan FileInfo, <-chan error) {
	fileChan := make(chan FileInfo)
	errChan := make(chan error, 1)

	go func() {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				return dirWalk(info)
			}

			if predicate(info, path) {
				fileChan <- FileInfo{Path: path, Info: info}
			}

			return nil
		})

		if err != nil {
			errChan <- err
		}

		// 先关闭 fileChan，之后再 errChan
		close(fileChan)
		close(errChan)
	}()

	return fileChan, errChan
}

var skippedDirs = map[string]struct{}{
	"target":  primitive.Empty,
	"mappers": primitive.Empty, // mybatis mapper xml files
	"test":    primitive.Empty,
}

func walkDir(info os.FileInfo) error {
	name := info.Name()

	if _, shouldSkip := skippedDirs[name]; shouldSkip {
		return filepath.SkipDir
	}

	if len(name) > 2 && strings.HasPrefix(name, ".") { // .git, .idea
		return filepath.SkipDir
	}

	return nil
}
