package mybatis

import (
	"regexp"
	"strings"

	"github.com/beevik/etree"
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

type MyBatisProcessor struct {
	fragments SQLFragments
}

func NewMyBatisProcessor() *MyBatisProcessor {
	return &MyBatisProcessor{fragments: make(SQLFragments)}
}

func (mp *MyBatisProcessor) ExtractSQLFragments(root *etree.Element) {
	for _, elem := range root.ChildElements() {
		if elem.Tag == "sql" {
			id := elem.SelectAttrValue("id", "")
			if id != "" {
				mp.fragments[id] = mp.extractRawSQL(elem)
			}
		}
	}
}

func (mp *MyBatisProcessor) PreprocessStmt(stmt *etree.Element) (rawSQL, preprocessedSQL, stmtID string) {
	rawSQL = mp.extractRawSQL(stmt)
	preprocessedSQL = mp.preprocessRawSQL(rawSQL)
	stmtID = stmt.SelectAttrValue("id", "")
	return
}

func (mp *MyBatisProcessor) extractRawSQL(elem *etree.Element) string {
	var sql strings.Builder
	mp.extractSQLRecursive(elem, &sql)
	return strings.TrimSpace(sql.String())
}

func (mp *MyBatisProcessor) extractSQLRecursive(elem *etree.Element, sql *strings.Builder) {
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			sql.WriteString(v.Data)
		case *etree.Element:
			if v.Tag == "![CDATA[" {
				sql.WriteString(v.Text())
			} else {
				sql.WriteString("<" + v.Tag)
				for _, attr := range v.Attr {
					sql.WriteString(" " + attr.Key + "=\"" + attr.Value + "\"")
				}
				sql.WriteString(">")
				mp.extractSQLRecursive(v, sql)
				sql.WriteString("</" + v.Tag + ">")
			}
		}
	}
}

func (mp *MyBatisProcessor) preprocessRawSQL(rawSQL string) string {
	// 处理 <include> 标签
	rawSQL = includeRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		refID := includeRegex.FindStringSubmatch(match)[1]
		return mp.fragments[refID]
	})

	// 处理 <where> 标签
	rawSQL = whereRegex.ReplaceAllString(rawSQL, "WHERE 1=1 $1")

	// 处理 <choose> 标签
	rawSQL = chooseRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
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
	rawSQL = foreachRegex.ReplaceAllString(rawSQL, "(...)")

	// 替换变量
	rawSQL = dollarVarRegex.ReplaceAllString(rawSQL, "''")
	rawSQL = hashVarRegex.ReplaceAllString(rawSQL, "?")

	// 移除所有剩余的 XML 标签
	rawSQL = tagRegex.ReplaceAllString(rawSQL, "")

	// 清理多余的空白字符
	rawSQL = strings.Join(strings.Fields(rawSQL), " ")

	return strings.TrimSpace(rawSQL)
}
