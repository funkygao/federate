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

	JavaAST = fs.ParseTemplateToString("templates/prompt/java_ast.md", promptData)
	JavaAPI = fs.ParseTemplateToString("templates/prompt/api_summary_zh.md", promptData)

	WMSMyBatisCN   = fs.ParseTemplateToString("templates/prompt/mybatis_mapper_zh.md", promptData)
	WMSMyBatisEN   = fs.ParseTemplateToString("templates/prompt/mybatis_mapper.md", promptData)
	WMSMyBatisMini = fs.ParseTemplateToString("templates/prompt/mybatis_mapper_mini.md", promptData)

	rules = []Rule{
		Rule{
			Name:   "WMS-MyBatisMapper分析-中文",
			Prompt: WMSMyBatisCN,
		},
		Rule{
			Name:   "WMS-API分析",
			Prompt: fs.ParseTemplateToString("templates/prompt/api_summary_zh.md", promptData),
		},
		Rule{
			Name:   "refactor",
			Prompt: fs.ParseTemplateToString("templates/prompt/refactor.md", promptData),
		},
	}
)
