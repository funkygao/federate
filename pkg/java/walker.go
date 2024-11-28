package java

import (
	"os"
	"path/filepath"
)

type fileHandler func(path string) error
type walkFilter func(info os.FileInfo, path string) bool

func ListXMLFiles(root string) ([]string, error) {
	return listFiles(root, walkXML)
}

func ListResourceFiles(root string) ([]string, error) {
	return listFiles(root, walkResources)
}

func ListJavaMainSourceFiles(root string) ([]string, error) {
	return listFiles(root, walkCode)
}

func listFiles(root string, walkFunc func(string, fileHandler) error) ([]string, error) {
	var files []string
	err := walkFunc(root, func(path string) error {
		files = append(files, path)
		return nil
	})
	return files, err
}

func walkCode(root string, handler fileHandler) error {
	return walk(root, nil, IsJavaMainSource, handler)
}

func walkResources(root string, handler fileHandler) error {
	return walk(root, nil, IsResourceFile, handler)
}

func walkXML(root string, handler fileHandler) error {
	return walk(root, nil, IsXML, handler)
}

func walk(root string, dirFilter walkFilter, fileFilter walkFilter, fileHandler fileHandler) error {
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
