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
	tagRegex       = regexp.MustCompile(`<[^>]+>`)
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
		if whenMatch := whenRegex.FindStringSubmatch(match); len(whenMatch) > 1 {
			return whenMatch[1]
		}
		if otherwiseMatch := otherwiseRegex.FindStringSubmatch(match); len(otherwiseMatch) > 1 {
			return otherwiseMatch[1]
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

	return strings.TrimSpace(xmlContent)
}
