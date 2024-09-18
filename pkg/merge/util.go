package merge

import (
	"io"
	"io/fs"
	"log"
	"os"
	"strings"

	"github.com/sergi/go-diff/diffmatchpatch"
)

func CopyFile(sourceFile, targetFile string) error {
	source, err := os.Open(sourceFile)
	if err != nil {
		return err
	}
	defer source.Close()

	target, err := os.Create(targetFile)
	if err != nil {
		return err
	}
	defer target.Close()

	_, err = io.Copy(target, source)
	return err
}

func handleFileErr(componentName, filePath string, err error) error {
	if _, ok := err.(*fs.PathError); ok {
		return nil
	}
	return err
}

func showRelevantDiffs(oldText, newText string) {
	dmp := diffmatchpatch.New()
	oldLines := strings.Split(oldText, "\n")
	newLines := strings.Split(newText, "\n")

	for i := 0; i < len(oldLines) || i < len(newLines); i++ {
		var oldLine, newLine string
		if i < len(oldLines) {
			oldLine = oldLines[i]
		}
		if i < len(newLines) {
			newLine = newLines[i]
		}

		if oldLine != newLine {
			diffs := dmp.DiffMain(oldLine, newLine, false)
			if len(diffs) == 1 && diffs[0].Type == diffmatchpatch.DiffDelete {
				// 完全删除的行
				log.Printf("\033[31m-%s\033[0m", oldLine)
			} else if len(diffs) == 1 && diffs[0].Type == diffmatchpatch.DiffInsert {
				// 完全新增的行
				log.Printf("\033[32m+%s\033[0m", newLine)
			} else {
				// 部分修改的行
				prettyDiff := dmp.DiffPrettyText(diffs)
				prettyDiff = prettyDiff
				log.Print(prettyDiff)
			}
		}
	}
}
