package prompt

import (
	"federate/internal/fs"
)

type Rule struct {
	Name   string
	Prompt string
}

var rules = []Rule{
	Rule{
		Name:   "WMS-MyBatisMapper分析-中文",
		Prompt: fs.ParseTemplateToString("templates/prompt/mybatis_mapper_zh.md", nil),
	},
	Rule{
		Name:   "WMS-API分析",
		Prompt: fs.ParseTemplateToString("templates/prompt/api_summary_zh.md", nil),
	},
	Rule{
		Name:   "WMS-MyBatisMapper分析",
		Prompt: fs.ParseTemplateToString("templates/prompt/mybatis_mapper.md", nil),
	},
	Rule{
		Name:   "refactor",
		Prompt: fs.ParseTemplateToString("templates/prompt/refactor.md", nil),
	},
}
