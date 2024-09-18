package corpus

import (
	"embed"
	"fmt"
	"io/fs"
	"strings"
)

//go:embed brown_corpus/*
var brownCorpus embed.FS

// GetCorpus returns the entire Brown Corpus as a single string
func GetCorpus() string {
	var builder strings.Builder

	err := fs.WalkDir(brownCorpus, "brown_corpus", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			content, err := brownCorpus.ReadFile(path)
			if err != nil {
				return fmt.Errorf("failed to read file %s: %v", path, err)
			}
			builder.Write(content)
			builder.WriteString("\n")
		}
		return nil
	})

	if err != nil {
		panic(fmt.Sprintf("Failed to read corpus: %v", err))
	}

	return builder.String()
}
