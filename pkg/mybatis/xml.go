package mybatis

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/beevik/etree"
)

var (
	hashPlaceHolder = regexp.MustCompile(`#\{[^}]+\}`)
	spaceRegex      = regexp.MustCompile(`\s+`)

	ErrNotMapperXML = fmt.Errorf("root element 'mapper' not found")
)

type SqlFragmentRef struct {
	Refid  string
	StmtID string
}

type Statement struct {
	Filename     string
	ID           string
	Type         string
	SQL          string
	ParseableSQL string
}

type XMLMapperBuilder struct {
	Filename         string
	Root             *etree.Element
	Namespace        string
	Statements       map[string]*Statement
	SqlFragments     map[string]string
	UnknownFragments []SqlFragmentRef
}

func NewXMLMapperBuilder(filename string) *XMLMapperBuilder {
	return &XMLMapperBuilder{
		Filename:         filename,
		Statements:       make(map[string]*Statement),
		SqlFragments:     make(map[string]string),
		UnknownFragments: []SqlFragmentRef{},
	}
}

func (b *XMLMapperBuilder) BaseName() string {
	return filepath.Base(b.Filename)
}

func (b *XMLMapperBuilder) ParseString(xmlContent string) (map[string]*Statement, error) {
	doc := etree.NewDocument()
	if err := doc.ReadFromString(xmlContent); err != nil {
		return nil, err
	}

	return b.doParse(doc)
}

func (b *XMLMapperBuilder) Parse() (map[string]*Statement, error) {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(b.Filename); err != nil {
		return nil, err
	}

	return b.doParse(doc)
}

func (b *XMLMapperBuilder) doParse(doc *etree.Document) (map[string]*Statement, error) {
	root := doc.SelectElement("mapper")
	if root == nil {
		return nil, ErrNotMapperXML
	}

	b.Root = root
	b.Namespace = root.SelectAttrValue("namespace", "")

	// 首先解析所有的 <sql> 标签
	for _, sqlElem := range root.SelectElements("sql") {
		if id := sqlElem.SelectAttrValue("id", ""); id != "" {
			b.SqlFragments[id] = strings.TrimSpace(b.processSqlFragment(sqlElem))
		}
	}

	// 然后解析其他语句
	for _, elem := range root.ChildElements() {
		switch elem.Tag {
		case "select", "insert", "update", "delete", "replace":
			stmt := &Statement{
				ID:   elem.SelectAttrValue("id", ""),
				Type: elem.Tag,
				SQL:  b.processDynamicSql(elem),
			}
			b.Statements[b.Namespace+"."+stmt.ID] = stmt
		}
	}

	// Post process SQL
	for _, stmt := range b.Statements {
		stmt.Filename = b.Filename
		stmt.ParseableSQL = b.postProcessSQL(stmt.SQL)
	}

	return b.Statements, nil
}

func (b *XMLMapperBuilder) processSqlFragment(elem *etree.Element) string {
	var sql strings.Builder
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			sql.WriteString(v.Data)
		case *etree.Element:
			if v.Tag == "include" {
				refid := v.SelectAttrValue("refid", "")
				if fragment, ok := b.SqlFragments[refid]; ok {
					sql.WriteString(fragment)
				}
			} else {
				sql.WriteString(b.processDynamicSql(v))
			}
		}
	}
	return sql.String()
}

// recursively process SQL
func (b *XMLMapperBuilder) processDynamicSql(elem *etree.Element) string {
	var sql strings.Builder

	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			sql.WriteString(v.Data)

		case *etree.Element:
			switch v.Tag {
			case "if":
				sql.WriteString(b.processDynamicSql(v))

			case "choose":
				whenElem := v.SelectElement("when")
				if whenElem != nil {
					sql.WriteString(b.processDynamicSql(whenElem))
				}

			case "trim", "where", "set":
				sql.WriteString(b.processWhereTrimSet(v))

			case "foreach":
				sql.WriteString(b.processForeach(v))

			case "include":
				refid := v.SelectAttrValue("refid", "")
				if sqlFragment, ok := b.SqlFragments[refid]; ok {
					sql.WriteString(sqlFragment)
				} else if refid != "" {
					b.UnknownFragments = append(b.UnknownFragments, SqlFragmentRef{Refid: refid, StmtID: elem.SelectAttrValue("id", "")})
				}
			}
		}
	}

	return sql.String()
}

func (b *XMLMapperBuilder) processWhereTrimSet(elem *etree.Element) string {
	content := b.processDynamicSql(elem)
	content = strings.TrimSpace(content)

	if elem.Tag == "where" && strings.HasPrefix(content, "AND ") {
		content = content[4:]
	}

	if content != "" {
		if elem.Tag == "where" {
			return "WHERE " + content
		} else if elem.Tag == "set" {
			return "SET " + content
		}
	}

	return content
}

func (b *XMLMapperBuilder) processForeach(elem *etree.Element) string {
	var result strings.Builder
	open := elem.SelectAttrValue("open", "")
	close := elem.SelectAttrValue("close", "")
	separator := elem.SelectAttrValue("separator", "")

	result.WriteString(open)

	isInsert := elem.Parent().Tag == "insert"

	if isInsert {
		// 批量插入场景
		result.WriteString("/* FOREACH_START */")
		innerContent := b.processDynamicSql(elem)
		// 替换所有 #{...} 为 ?
		innerContent = hashPlaceHolder.ReplaceAllString(innerContent, "?")
		result.WriteString(innerContent)
		result.WriteString("/* FOREACH_END */")

		// 添加 FOREACH_ITEM 注释，但不添加额外的逗号
		if separator != "" {
			result.WriteString("/* FOREACH_ITEM */")
		}
	} else {
		// 其他场景（如 SELECT 中的 IN 子句）
		result.WriteString("?")
		if separator != "" {
			result.WriteString(separator)
			result.WriteString("?")
		}
	}

	result.WriteString(close)

	return result.String()
}

func (b *XMLMapperBuilder) postProcessSQL(sql string) string {
	sql = removeCDATA(sql)
	sql = replaceMybatisPlaceholders(sql)
	sql = removeExtraWhitespace(sql)
	return strings.TrimSpace(sql)
}

func removeCDATA(sql string) string {
	return strings.ReplaceAll(strings.ReplaceAll(sql, "<![CDATA[", ""), "]]>", "")
}

func removeExtraWhitespace(sql string) string {
	// Replace multiple spaces with a single space
	sql = spaceRegex.ReplaceAllString(sql, " ")
	return sql
}

func replaceMybatisPlaceholders(sql string) string {
	return hashPlaceHolder.ReplaceAllString(sql, "?")
}
