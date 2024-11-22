package java

import (
	"path/filepath"
)

type FileHandler func(path string) error

type WalkFilter func(path string, info os.FileInfo) bool

func WalkCode(root string, handler FileHandler) error {
	return Walk(root, nil, IsJavaMainSourc, handler)
}

func WalkXML(root string, handler FileHandler) error {
	return Walk(root, nil, IsXml, handler)
}

func Walk(root string, dirFilter WalkFilter, fileFilter WalkFilter, fileHandler FileHandler) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			if dirFilter != nil && !dirFilter(path, info) {
				return filepath.SkipDir
			} else {
				return nil
			}
		} else if fileFilter != nil && !fileFilter(path, info) {
			return nil
		}

		// path is file, not dir
		return fileHandler(path)
	})
}
