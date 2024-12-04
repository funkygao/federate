package diff

import (
	"log"
	"strings"

	"github.com/sergi/go-diff/diffmatchpatch"
)

// Render diff in unified mode instead of side by side.
// Limitation: Only works if no line Insert/Delete.
func RenderUnifiedDiff(oldText, newText string) {
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
				log.Print(prettyDiff)
			}
		}
	}
}

func ShowDiffLineByLine(oldText, newText string) {
	dmp := diffmatchpatch.New()

	// 计算差异
	diffs := dmp.DiffMain(oldText, newText, false)

	// 清理差异
	diffs = dmp.DiffCleanupSemantic(diffs)

	// 构建输出
	var currentLine strings.Builder

	flushLine := func() {
		if currentLine.Len() > 0 {
			line := currentLine.String()
			if strings.Contains(line, "\033") {
				// 有改动
				log.Printf("%s", line)
			}
			currentLine.Reset()
		}
	}

	for _, diff := range diffs {
		diffLines := strings.Split(diff.Text, "\n")
		for i, diffLine := range diffLines {
			switch diff.Type {
			case diffmatchpatch.DiffInsert:
				currentLine.WriteString("\033[32m" + diffLine + "\033[0m")
			case diffmatchpatch.DiffDelete:
				currentLine.WriteString("\033[31m" + diffLine + "\033[0m")
			case diffmatchpatch.DiffEqual:
				currentLine.WriteString(diffLine)
			}

			if i < len(diffLines)-1 {
				flushLine()
			}
		}
	}
}
