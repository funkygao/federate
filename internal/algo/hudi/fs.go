package main

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

type FileSystem interface {
	Write(path string, data []byte) error
	Read(path string) ([]byte, error)
	List(path string) ([]string, error)
	Delete(path string) error
}

type LocalFileSystem struct {
	rootDir string
}

func NewLocalFileSystem(rootDir string) (*LocalFileSystem, error) {
	if err := os.MkdirAll(rootDir, 0755); err != nil {
		return nil, err
	}
	return &LocalFileSystem{rootDir: rootDir}, nil
}

func (fs *LocalFileSystem) Write(path string, data []byte) error {
	fullPath := filepath.Join(fs.rootDir, path)
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return ioutil.WriteFile(fullPath, data, 0644)
}

func (fs *LocalFileSystem) Read(path string) ([]byte, error) {
	fullPath := filepath.Join(fs.rootDir, path)
	return ioutil.ReadFile(fullPath)
}

func (fs *LocalFileSystem) List(path string) ([]string, error) {
	fullPath := filepath.Join(fs.rootDir, path)
	files, err := ioutil.ReadDir(fullPath)
	if err != nil {
		return nil, err
	}
	var result []string
	for _, file := range files {
		result = append(result, file.Name())
	}
	return result, nil
}

func (fs *LocalFileSystem) Delete(path string) error {
	fullPath := filepath.Join(fs.rootDir, path)
	return os.Remove(fullPath)
}
