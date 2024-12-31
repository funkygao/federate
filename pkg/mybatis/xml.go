package mybatis

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/beevik/etree"
)

var (
	hashPlaceHolder = regexp.MustCompile(`#\{[^}]+\}`)
	ErrNotMapperXML = fmt.Errorf("root element 'mapper' not found")
)

type Statement struct {
	Filename     string
	ID           string
	Type         string
	SQL          string
	ParseableSQL string
}

type XMLMapperBuilder struct {
	Filename     string
	Namespace    string
	Statements   map[string]*Statement
	SqlFragments map[string]string
}

func NewXMLMapperBuilder(filename string) *XMLMapperBuilder {
	return &XMLMapperBuilder{
		Filename:     filename,
		Statements:   make(map[string]*Statement),
		SqlFragments: make(map[string]string),
	}
}

func (b *XMLMapperBuilder) Parse() (map[string]*Statement, error) {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(b.Filename); err != nil {
		return nil, err
	}

	root := doc.SelectElement("mapper")
	if root == nil {
		return nil, ErrNotMapperXML
	}

	b.Namespace = root.SelectAttrValue("namespace", "")

	// Parse SQL fragments
	for _, sqlElem := range root.SelectElements("sql") {
		id := sqlElem.SelectAttrValue("id", "")
		if id != "" {
			b.SqlFragments[id] = b.processDynamicSql(sqlElem)
		}
	}

	// Parse statements
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

	for _, stmt := range b.Statements {
		stmt.Filename = b.Filename
		stmt.ParseableSQL = b.postProcessSQL(stmt.SQL)
	}

	return b.Statements, nil
}

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

	if separator != "" {
		// If there's a separator, we'll add two placeholders to represent multiple items
		result.WriteString("?")
		result.WriteString(separator)
		result.WriteString("?")
	} else {
		// If there's no separator, we'll just add one placeholder
		result.WriteString("?")
	}

	result.WriteString(close)

	return result.String()
}

func (b *XMLMapperBuilder) postProcessSQL(sql string) string {
	sql = removeCDATA(sql)
	sql = replaceMybatisPlaceholders(sql)
	return strings.TrimSpace(sql)
}

func removeCDATA(sql string) string {
	return strings.ReplaceAll(strings.ReplaceAll(sql, "<![CDATA[", ""), "]]>", "")
}

func replaceMybatisPlaceholders(sql string) string {
	return hashPlaceHolder.ReplaceAllString(sql, "?")
}
