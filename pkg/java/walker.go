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

func WalkResources(root string, handler FileHandler) error {
	return Walk(root, nil, IsResourceFile, handler)
}

func WalkXML(root string, handler FileHandler) error {
	return Walk(root, nil, IsXML, handler)
}

func ListXMLFiles(root string) ([]string, error) {
	return listFiles(root, WalkXML)
}

func ListResourceFiles(root string) ([]string, error) {
	return listFiles(root, WalkResources)
}

func ListJavaMainSourceFiles(root string) ([]string, error) {
	return listFiles(root, WalkCode)
}

func Walk(root string, dirFilter WalkFilter, fileFilter WalkFilter, fileHandler FileHandler) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if dirFilter != nil && !dirFilter(info, path) {
				return filepath.SkipDir
			}
			return nil
		}

		if fileFilter != nil && !fileFilter(info, path) {
			return nil
		}

		// path is file, not dir
		return fileHandler(path)
	})
}

func listFiles(root string, walkFunc func(string, FileHandler) error) ([]string, error) {
	var files []string
	err := walkFunc(root, func(path string) error {
		files = append(files, path)
		return nil
	})
	return files, err
}
