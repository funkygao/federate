package ast

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"federate/pkg/java"
)

type fileInfo struct {
	Path    string
	Content string
}

type javaFileProducer struct {
	parser *javaParser
}

var fileChannelBufferSize = runtime.NumCPU() * 2

func (p *javaFileProducer) Produce(dir string) <-chan fileInfo {
	filesChan := make(chan fileInfo, fileChannelBufferSize)

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
					log.Println(path)
				}

				// 如果 channel 已满，会自动阻塞
				filesChan <- fileInfo{Path: path, Content: string(content)}
			}
			return nil
		})
	}()

	return filesChan
}
