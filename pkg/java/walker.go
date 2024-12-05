package java

import (
	"os"
	"path/filepath"
)

type FilePredicate func(info os.FileInfo, path string) bool

func ListXMLFiles(root string) ([]string, error) {
	return ListFiles(root, IsXML)
}

func ListResourceFiles(root string) ([]string, error) {
	return ListFiles(root, IsResourceFile)
}

func ListJavaMainSourceFiles(root string) ([]string, error) {
	return ListFiles(root, IsJavaMainSource)
}

func ListFiles(root string, predicate FilePredicate) (files []string, err error) {
	resultChan := make(chan string, 2000)
	errChan := make(chan error, 1)

	go func() {
		if err := parallelWalk(root, predicate, resultChan, errChan); err != nil {
			errChan <- err
		}

		close(resultChan)
		close(errChan)
	}()

	for file := range resultChan {
		files = append(files, file)
	}

	if err = <-errChan; err != nil {
		return
	}

	return
}

func parallelWalk(root string, predicate FilePredicate, resultChan chan<- string, errChan chan<- error) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if ShouldSkipDir(info) {
				return filepath.SkipDir
			}
			return nil
		}

		if predicate(info, path) {
			resultChan <- path
		}

		return nil
	})
}
