package ast

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"federate/pkg/java"
)

// FileInfo 包含要解析的文件信息
type FileInfo struct {
	Path    string
	Content string
}

// Producer 定义了生产者接口
type Producer interface {
	Produce(dir string) (<-chan FileInfo, <-chan error)
}

type JavaFileProducer struct{}

func (p *JavaFileProducer) Produce(dir string) (<-chan FileInfo, <-chan error) {
	filesChan := make(chan FileInfo)
	errorsChan := make(chan error)

	go func() {
		defer close(filesChan)
		defer close(errorsChan)

		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if java.IsJavaMainSource(info, path) {
				content, err := ioutil.ReadFile(path)
				if err != nil {
					errorsChan <- fmt.Errorf("error reading file %s: %v", path, err)
					return nil
				}
				filesChan <- FileInfo{Path: path, Content: string(content)}
			}
			return nil
		})

		if err != nil {
			errorsChan <- err
		}
	}()

	return filesChan, errorsChan
}
