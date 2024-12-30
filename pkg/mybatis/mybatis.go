package mybatis

import (
	"regexp"
	"strings"
)

type SQLFragments map[string]string

var (
	RootTag = "mapper"

	includeRegex   = regexp.MustCompile(`<include\s+refid="([^"]+)"\s*/>`)
	whereRegex     = regexp.MustCompile(`<where>(?s)(.*?)</where>`)
	chooseRegex    = regexp.MustCompile(`<choose>(?s)(.*?)</choose>`)
	whenRegex      = regexp.MustCompile(`<when[^>]*>(?s)(.*?)</when>`)
	otherwiseRegex = regexp.MustCompile(`<otherwise>(?s)(.*?)</otherwise>`)
	foreachRegex   = regexp.MustCompile(`<foreach[^>]*>(?s)(.*?)</foreach>`)
	dollarVarRegex = regexp.MustCompile(`\$\{([^}]+)\}`)
	hashVarRegex   = regexp.MustCompile(`#\{([^}]+)\}`)
	tagRegex       = regexp.MustCompile(`</?[^>]+>`)
)

func preprocessMyBatisXML(xmlContent string, fragments SQLFragments) string {
	// 处理 <include> 标签
	xmlContent = includeRegex.ReplaceAllStringFunc(xmlContent, func(match string) string {
		refID := includeRegex.FindStringSubmatch(match)[1]
		return fragments[refID]
	})

	// 处理 <where> 标签
	xmlContent = whereRegex.ReplaceAllString(xmlContent, "WHERE 1=1 $1")

	// 处理 <choose> 标签
	xmlContent = chooseRegex.ReplaceAllStringFunc(xmlContent, func(match string) string {
		whenMatches := whenRegex.FindAllStringSubmatch(match, -1)
		for _, whenMatch := range whenMatches {
			if len(whenMatch) > 1 {
				return strings.TrimSpace(whenMatch[1])
			}
		}
		if otherwiseMatch := otherwiseRegex.FindStringSubmatch(match); len(otherwiseMatch) > 1 {
			return strings.TrimSpace(otherwiseMatch[1])
		}
		return ""
	})

	// 处理 <foreach> 标签
	xmlContent = foreachRegex.ReplaceAllString(xmlContent, "(...)")

	// 替换变量
	xmlContent = dollarVarRegex.ReplaceAllString(xmlContent, "''")
	xmlContent = hashVarRegex.ReplaceAllString(xmlContent, "?")

	// 移除所有剩余的 XML 标签
	xmlContent = tagRegex.ReplaceAllString(xmlContent, "")

	// 清理多余的空白字符
	xmlContent = strings.Join(strings.Fields(xmlContent), " ")

	return strings.TrimSpace(xmlContent)
}
