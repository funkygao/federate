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
	SqlFragments     map[string]*etree.Element
	UnknownFragments []SqlFragmentRef
	nodeHandlers     map[string]NodeHandler
}

func NewXMLMapperBuilder(filename string) *XMLMapperBuilder {
	return &XMLMapperBuilder{
		Filename:         filename,
		Statements:       make(map[string]*Statement),
		SqlFragments:     make(map[string]*etree.Element),
		UnknownFragments: []SqlFragmentRef{},
		nodeHandlers:     newNodeHandlers(),
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
			b.SqlFragments[id] = b.processSqlFragment(sqlElem)
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
				Raw:      elementToString(elem),
				SQL:      b.processDynamicSql(elem),
			}
			b.Statements[b.Namespace+"."+stmt.ID] = stmt
		}
	}

	// Post process SQL
	for _, stmt := range b.Statements {
		stmt.SQL = b.postProcessSQL(stmt.SQL)
		if Verbosity > 2 {
			color.Yellow("%s %s", b.BaseName(), stmt.ID)
			log.Println(stmt.SQL)
		}
	}

	return b.Statements, nil
}

func elementToString(elem *etree.Element) string {
	var buf bytes.Buffer
	var s etree.WriteSettings
	elem.WriteTo(&buf, &s)
	return buf.String()
}

func (b *XMLMapperBuilder) processSqlFragment(elem *etree.Element) *etree.Element {
	fragment := etree.NewElement("fragment")
	for _, child := range elem.Child {
		switch t := child.(type) {
		case *etree.Element:
			fragment.AddChild(t.Copy())
		case *etree.Comment:
			newComment := &etree.Comment{Data: t.Data}
			fragment.AddChild(newComment)
		case *etree.CharData:
			newCharData := &etree.CharData{Data: t.Data}
			fragment.AddChild(newCharData)
		case *etree.Directive:
			newDirective := &etree.Directive{Data: t.Data}
			fragment.AddChild(newDirective)
		case *etree.ProcInst:
			newProcInst := &etree.ProcInst{Target: t.Target, Inst: t.Inst}
			fragment.AddChild(newProcInst)
		default:
			// 如果需要，处理其他类型的节点
		}
	}
	return fragment
}

// 递归地处理 SQL
func (b *XMLMapperBuilder) processDynamicSql(elem *etree.Element) string {
	context := &DynamicContext{}
	b.parseDynamicTags(elem, context)
	return context.sql.String()
}

func (b *XMLMapperBuilder) parseDynamicTags(elem *etree.Element, context *DynamicContext) {
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			context.sql.WriteString(v.Data)
		case *etree.Element:
			handler, exists := b.nodeHandlers[v.Tag]
			if exists {
				handler.handle(b, v, context)
			} else if v.Tag == "include" {
				b.handleInclude(v, context)
			}
		}
	}
}

func (b *XMLMapperBuilder) handleInclude(elem *etree.Element, context *DynamicContext) {
	refid := elem.SelectAttrValue("refid", "")
	if fragmentElem, ok := b.SqlFragments[refid]; ok {
		b.parseDynamicTags(fragmentElem, context)
	} else if refid != "" {
		b.UnknownFragments = append(b.UnknownFragments, SqlFragmentRef{Refid: refid, StmtID: elem.SelectAttrValue("id", "")})
	}
}

func (b *XMLMapperBuilder) postProcessSQL(sql string) string {
	sql = removeCDATA(sql)
	sql = replaceMybatisPlaceholders(sql)
	sql = removeExtraWhitespace(sql)

	// 处理多个 ORDER BY 子句
	orderByParts := strings.Split(sql, "ORDER BY")
	if len(orderByParts) > 1 {
		sql = orderByParts[0]
		var orderByClauses []string
		for _, part := range orderByParts[1:] {
			orderByClauses = append(orderByClauses, strings.TrimSpace(part))
		}
		sql += "ORDER BY " + strings.Join(orderByClauses, ", ")
	}

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
