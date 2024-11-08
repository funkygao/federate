package ast

import (
	"io/ioutil"
	"log"
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

type JavaFileProducer struct {
	parser *javaParser
}

const fileChannelBufferSize = 500

func (p *JavaFileProducer) Produce(dir string) <-chan FileInfo {
	filesChan := make(chan FileInfo, fileChannelBufferSize)

	go func() {
		defer close(filesChan)

		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				log.Printf("%s: %v", path, err)
				return filepath.SkipDir
			}

			if java.IsJavaMainSource(info, path) {
				content, err := ioutil.ReadFile(path)
				if err != nil {
					log.Printf("Error reading %s: %v", path, err)
					return nil
				}

				if p.parser.debug {
					log.Printf("Parsing %s", path)
				}

				// 如果 channel 已满，会自动阻塞
				filesChan <- FileInfo{Path: path, Content: string(content)}
			}
			return nil
		})
	}()

	return filesChan
}
