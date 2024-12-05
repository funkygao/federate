package java

import (
	"fmt"
	"os"
	"path/filepath"
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

func ListFilesAsync(root string, predicate FilePredicate) (<-chan FileInfo, <-chan error) {
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

func ProcessFiles(fileChan <-chan FileInfo, errChan <-chan error, processor func(FileInfo) error) error {
	// ListFiles 时，close chan 顺序保证：fileChan -> errChan
	for fileInfo := range fileChan {
		if err := processor(fileInfo); err != nil {
			return fmt.Errorf("error processing file %s: %w", fileInfo.Path, err)
		}
	}

	// fileChan 已关闭，检查 errChan
	if err := <-errChan; err != nil {
		return fmt.Errorf("error listing files: %w", err)
	}

	return nil
}
