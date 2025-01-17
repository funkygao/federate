package ast

import (
	"fmt"
	"log"

	"federate/pkg/primitive"
	"federate/pkg/prompt"
	"federate/pkg/tabular"
	"github.com/fatih/color"
)

type SectionReporter func() (empty bool)

func (i *Info) ShowReport() {
	i.PrepareData()

	if Web {
		i.StartWebServer("8080")
		return
	}

	if GeneratePrompt {
		i.logger = prompt.NewPromptLogger()
		i.logger.AddPrompt("# The Detailed Java AST Report\n\n")
		i.logger.Start()
		defer i.logger.Stop()
	}

	sections := []SectionReporter{
		// overview
		i.showOverviewReport,

		// clusters
		i.showInterfacesReport,
		i.showInheritanceReport,
		i.showClusterRelationships,
		i.showCompositionReport,

		// computing/logic intensive
		i.showComplexConditionsReport,
		i.showComplexLoopsReport,
		i.showReflectionReport,

		// TODO: try catch, concurrency，哪个类 if/switch 最多

		// FP
		i.showFunctionalUsageReport,
		i.showLambdaReport,

		// Transaction
		i.showTransactionReport,
	}

	for i, section := range sections {
		if empty := section(); !empty && i != len(sections)-1 {
			log.Println()
		}
	}

	if GeneratePrompt {
		i.logger.AddPrompt(prompt.JavaAST)
	}
}

func (i *Info) writeSectionHeader(format string, v ...any) {
	title := fmt.Sprintf(format, v...)
	color.Magenta(title)

	if GeneratePrompt {
		i.logger.AddPrompt("\n## %s\n\n", title)
	}
}

func (i *Info) writeSectionBody(content func()) {
	if GeneratePrompt {
		i.logger.AddPrompt("```\n")
		content()
		i.logger.AddPrompt("```\n\n")
	} else {
		content()
	}
}

func (i *Info) showNameCountSection(title string, namesHeader []string, nameCounts ...[]primitive.NameCount) {
	var headers []string
	for _, name := range namesHeader {
		headers = append(headers, name, "Count")
	}

	maxLen := 0
	for _, nc := range nameCounts {
		if len(nc) > maxLen {
			maxLen = len(nc)
		}
	}

	cellData := make([][]string, maxLen)
	for i := range cellData {
		cellData[i] = make([]string, len(headers))
	}

	for colIndex, nc := range nameCounts {
		for rowIndex, item := range nc {
			if rowIndex < maxLen {
				cellData[rowIndex][colIndex*2] = item.Name
				cellData[rowIndex][colIndex*2+1] = fmt.Sprintf("%d", item.Count)
			}
		}
	}

	i.writeSectionHeader("Top %d %s", TopK, title)
	i.writeSectionBody(func() {
		tabular.Display(headers, cellData, false, -1)
	})
}
