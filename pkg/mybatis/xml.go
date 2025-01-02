package mybatis

import (
	"bytes"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/beevik/etree"
	"github.com/fatih/color"
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
	Filename string
	ID       string
	Type     string
	Raw      string
	SQL      string
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
				Filename: b.Filename,
				ID:       elem.SelectAttrValue("id", ""),
				Type:     elem.Tag,
				Raw:      b.elementText(elem),
				SQL:      b.processDynamicSql(elem),
			}
			b.Statements[b.Namespace+"."+stmt.ID] = stmt
		}
	}

	// Post process SQL
	for _, stmt := range b.Statements {
		stmt.SQL = b.postProcessSQL(stmt.SQL)
		if verbosity > 2 {
			color.Yellow("%s %s", b.BaseName(), stmt.ID)
			log.Println(stmt.SQL)
		}
	}

	return b.Statements, nil
}

func (b *XMLMapperBuilder) elementText(elem *etree.Element) string {
	var buf bytes.Buffer
	var s etree.WriteSettings
	elem.WriteTo(&buf, &s)
	return buf.String()
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

	// Handle prefixOverrides
	prefixOverrides := elem.SelectAttrValue("prefixOverrides", "")
	if prefixOverrides != "" {
		overrides := strings.Split(prefixOverrides, "|")
		for _, override := range overrides {
			override = strings.TrimSpace(override)
			if strings.HasPrefix(content, override) {
				content = strings.TrimPrefix(content, override)
				content = strings.TrimSpace(content)
				break // Remove only one matching override
			}
		}
	}

	// Handle suffixOverrides
	suffixOverrides := elem.SelectAttrValue("suffixOverrides", "")
	if suffixOverrides != "" {
		overrides := strings.Split(suffixOverrides, "|")
		for _, override := range overrides {
			override = strings.TrimSpace(override)
			if strings.HasSuffix(content, override) {
				content = strings.TrimSuffix(content, override)
				content = strings.TrimSpace(content)
				break // Remove only one matching override
			}
		}
	}

	// Handle prefix and suffix
	prefix := elem.SelectAttrValue("prefix", "")
	suffix := elem.SelectAttrValue("suffix", "")

	if content != "" {
		content = prefix + " " + content + " " + suffix
	}

	if elem.Tag == "where" && content != "" {
		return "WHERE " + content
	} else if elem.Tag == "set" && content != "" {
		return "SET " + content
	} else {
		return content
	}
}

func (b *XMLMapperBuilder) processForeach(elem *etree.Element) string {
	var result strings.Builder
	open := elem.SelectAttrValue("open", "")
	close := elem.SelectAttrValue("close", "")
	separator := elem.SelectAttrValue("separator", "")

	result.WriteString(open)

	isInsert := elem.Parent().Tag == "insert"

	// 处理内部的内容
	innerContent := b.processDynamicSql(elem)
	// 替换所有 #{...} 为 ?
	innerContent = hashPlaceHolder.ReplaceAllString(innerContent, "?")

	if isInsert {
		// 批量插入场景
		result.WriteString("/* FOREACH_START */")
		result.WriteString(innerContent)
		result.WriteString("/* FOREACH_END */")

		// 添加 FOREACH_ITEM 注释，但不添加额外的逗号
		if separator != "" {
			result.WriteString("/* FOREACH_ITEM */")
		}
	} else {
		// 非插入的场景，需要根据分隔符处理
		// 假设循环至少执行一次，我们用占位符代表多次执行
		// 为了测试的目的，我们构造一个示例输出
		simulatedLoop := []string{innerContent}
		if separator != "" {
			simulatedLoop = append(simulatedLoop, separator)
			simulatedLoop = append(simulatedLoop, innerContent)
		}
		result.WriteString(strings.Join(simulatedLoop, " "))
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
