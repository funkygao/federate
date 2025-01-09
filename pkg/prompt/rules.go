package prompt

import (
	"os"

	"federate/internal/fs"
)

type Rule struct {
	Name   string
	Prompt string
}

var (
	promptData = struct {
		Domain string
	}{
		Domain: os.Getenv("PROMPT_DOMAIN"),
	}

	rules = []Rule{
		Rule{
			Name:   "WMS-MyBatisMapper分析-中文",
			Prompt: fs.ParseTemplateToString("templates/prompt/mybatis_mapper_zh.md", promptData),
		},
		Rule{
			Name:   "WMS-API分析",
			Prompt: fs.ParseTemplateToString("templates/prompt/api_summary_zh.md", promptData),
		},
		Rule{
			Name:   "WMS-MyBatisMapper分析",
			Prompt: fs.ParseTemplateToString("templates/prompt/mybatis_mapper.md", promptData),
		},
		Rule{
			Name:   "refactor",
			Prompt: fs.ParseTemplateToString("templates/prompt/refactor.md", promptData),
		},
	}
)
