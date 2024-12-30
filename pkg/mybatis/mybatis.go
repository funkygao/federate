package mybatis

import (
	"regexp"
	"strings"

	"github.com/beevik/etree"
)

type SQLFragments map[string]string

var (
	RootTag = "mapper"

	includeRegex   = regexp.MustCompile(`<include\s+refid="([^"]+)"(?:\s*/>|[^>]*>\s*</include>)`)
	whereRegex     = regexp.MustCompile(`<where>(?s)(.*?)</where>`)
	chooseRegex    = regexp.MustCompile(`<choose>(?s)(.*?)</choose>`)
	whenRegex      = regexp.MustCompile(`<when[^>]*>(?s)(.*?)</when>`)
	ifRegex        = regexp.MustCompile(`<if[^>]*>(?s)(.*?)</if>`)
	otherwiseRegex = regexp.MustCompile(`<otherwise>(?s)(.*?)</otherwise>`)
	foreachRegex   = regexp.MustCompile(`<foreach[^>]*>(?s)(.*?)</foreach>`)
	dollarVarRegex = regexp.MustCompile(`\$\{([^}]+)\}`)
	hashVarRegex   = regexp.MustCompile(`#\{([^}]+)\}`)
	tagRegex       = regexp.MustCompile(`</?[^>]+>`)

	jsonOperatorRegex      = regexp.MustCompile(`(\w+)\s*->>\s*#\{[^}]+\}\s*=\s*#\{[^}]+\}`)
	variableInForeachRegex = regexp.MustCompile(`#\{[^}]+\}`)
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
			if id := elem.SelectAttrValue("id", ""); id != "" {
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

func (mp *MyBatisProcessor) processIncludes(sql string) string {
	for includeRegex.MatchString(sql) {
		sql = includeRegex.ReplaceAllStringFunc(sql, func(match string) string {
			refID := includeRegex.FindStringSubmatch(match)[1]
			return mp.processIncludes(mp.fragments[refID])
		})
	}
	return sql
}

func (mp *MyBatisProcessor) preprocessRawSQL(rawSQL string) string {
	// 处理 <include> 标签
	rawSQL = mp.processIncludes(rawSQL)

	// 处理 <where> 标签
	rawSQL = whereRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		inner := whereRegex.FindStringSubmatch(match)[1]
		return "WHERE " + strings.TrimSpace(inner)
	})

	// 处理 <choose> 标签
	rawSQL = chooseRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		whenMatches := whenRegex.FindAllStringSubmatch(match, -1)
		for _, whenMatch := range whenMatches {
			if len(whenMatch) > 1 {
				return "?"
			}
		}
		if otherwiseMatch := otherwiseRegex.FindStringSubmatch(match); len(otherwiseMatch) > 1 {
			return strings.TrimSpace(otherwiseMatch[1])
		}
		return "?"
	})

	// 处理 <if> 标签
	rawSQL = ifRegex.ReplaceAllString(rawSQL, "$1")

	// 处理 <foreach> 标签，保留 JSON 操作符
	rawSQL = foreachRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		innerContent := foreachRegex.FindStringSubmatch(match)[1]
		if strings.Contains(innerContent, "VALUES") || strings.Contains(innerContent, "values") {
			return "(?, ?, ?, ?), (?, ?, ?, ?)"
		}
		if strings.Contains(innerContent, "->>") {
			// 保留 JSON 操作符，替换变量，并保持括号
			return "(" + jsonOperatorRegex.ReplaceAllString(innerContent, "$1 ->> ? = ?") + ")"
		}
		return "(?)"
	})

	// 替换变量，保留 JSON 操作符
	rawSQL = jsonOperatorRegex.ReplaceAllString(rawSQL, "$1 ->> ? = ?")
	rawSQL = dollarVarRegex.ReplaceAllString(rawSQL, "?")
	rawSQL = hashVarRegex.ReplaceAllString(rawSQL, "?")

	// 移除所有剩余的 XML 标签
	rawSQL = tagRegex.ReplaceAllString(rawSQL, "")

	// 清理多余的空白字符
	rawSQL = strings.Join(strings.Fields(rawSQL), " ")

	return strings.TrimSpace(rawSQL)
}
