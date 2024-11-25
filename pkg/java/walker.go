package java

import (
	"os"
	"path/filepath"
)

type FileHandler func(path string) error

type WalkFilter func(info os.FileInfo, path string) bool

func WalkCode(root string, handler FileHandler) error {
	return Walk(root, nil, IsJavaMainSource, handler)
}

func WalkXML(root string, handler FileHandler) error {
	return Walk(root, nil, IsXml, handler)
}

func ListJavaMainSourceFiles(root string) ([]string, error) {
	var files []string
	err := WalkCode(root, func(path string) error {
		files = append(files, path)
		return nil
	})
	return files, err
}

func Walk(root string, dirFilter WalkFilter, fileFilter WalkFilter, fileHandler FileHandler) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if dirFilter != nil && !dirFilter(info, path) {
				return filepath.SkipDir
			} else {
				return nil
			}
		} else if fileFilter != nil && !fileFilter(info, path) {
			return nil
		}

		// path is file, not dir
		return fileHandler(path)
	})
}
