package java

import (
	"os"
	"path/filepath"
	"strings"
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
	fileChan := make(chan FileInfo)
	errChan := make(chan error, 1)

	go func() {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				return walkDir(info)
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

func walkDir(info os.FileInfo) error {
	name := info.Name()
	if name == "target" ||
		name == "test" ||
		(len(name) > 2 && strings.HasPrefix(name, ".")) { // .git, .idea
		return filepath.SkipDir
	}

	return nil
}
