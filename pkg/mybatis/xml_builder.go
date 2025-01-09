package mybatis

import (
	"bytes"
	"fmt"
	"log"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"federate/pkg/java"
	"github.com/beevik/etree"
	"github.com/fatih/color"
)

var (
	hashPlaceHolder = regexp.MustCompile(`#\{[^}]+\}`)
	spaceRegex      = regexp.MustCompile(`\s+`)

	GlobalSqlFragments     = make(map[string]*etree.Element)
	GlobalSqlFragmentUsage = make(map[string]int)

	ErrNotMapperXML = fmt.Errorf("root element 'mapper' not found")
)

type XMLMapperBuilder struct {
	Filename         string
	Root             *etree.Element
	Namespace        string
	Statements       map[string]*Statement
	UnknownFragments []SqlFragmentRef
	nodeHandlers     map[string]NodeHandler
}

func NewXMLMapperBuilder(filename string) *XMLMapperBuilder {
	return &XMLMapperBuilder{
		Filename:         filename,
		Statements:       make(map[string]*Statement),
		UnknownFragments: []SqlFragmentRef{},
		nodeHandlers:     newNodeHandlers(),
	}
}

func (b *XMLMapperBuilder) BaseName() string {
	return filepath.Base(b.Filename)
}

func (b *XMLMapperBuilder) Prepare() error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(b.Filename); err != nil {
		return err
	}

	root := doc.SelectElement("mapper")
	if root == nil {
		return ErrNotMapperXML
	}

	b.Root = root
	b.Namespace = root.SelectAttrValue("namespace", "")

	for _, sqlElem := range root.SelectElements("sql") {
		if id := sqlElem.SelectAttrValue("id", ""); id != "" {
			GlobalSqlFragments[b.Namespace+"."+id] = b.processSqlFragment(sqlElem)
		}
	}

	return nil
}

// Parse 只解析 CRUD 语句
func (b *XMLMapperBuilder) Parse() (map[string]*Statement, error) {
	if b.Root == nil {
		return nil, nil
	}

	for _, elem := range b.Root.ChildElements() {
		switch elem.Tag {
		case "select", "insert", "update", "delete", "replace":
			timeout, _ := strconv.Atoi(elem.SelectAttrValue("timeout", "0"))
			stmt := &Statement{
				Filename:      b.Filename,
				ID:            elem.SelectAttrValue("id", ""),
				ParameterType: java.ClassSimpleName(elem.SelectAttrValue("parameterType", "NULL")),
				ResultType:    java.ClassSimpleName(elem.SelectAttrValue("resultType", "NULL")),
				Tag:           elem.Tag,
				XMLText:       elementToString(elem),
				SQL:           b.processDynamicSql(elem),
				Timeout:       timeout,
				SubSQL:        []SQLet{},
			}
			b.Statements[b.Namespace+"."+stmt.ID] = stmt
		}
	}

	// Post process SQL
	for _, stmt := range b.Statements {
		stmt.SQL = b.postProcessSQL(stmt.SQL)
		if Verbosity > 4 {
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
	return context.String()
}

func (b *XMLMapperBuilder) parseDynamicTags(elem *etree.Element, context *DynamicContext) {
	for _, child := range elem.Child {
		switch v := child.(type) {
		case *etree.CharData:
			context.WriteString(v.Data)
		case *etree.Element:
			handler, exists := b.nodeHandlers[v.Tag]
			if exists {
				handler.handle(b, v, context)
			} else if v.Tag == "include" {
				b.handleInclude(v, context)
			} else if v.Tag == "bind" || v.Tag == "selectKey" {
				// <selectKey resultType="java.lang.Long"  keyProperty="id" keyColumn="id" order="AFTER">SELECT LAST_INSERT_ID()</selectKey>
				// <bind name="skuSize" value="sku.size"/>

				// ignore
			} else {
				log.Printf("Unknown tag: %s", v.Tag)
			}
		}
	}
}

func (b *XMLMapperBuilder) handleInclude(elem *etree.Element, context *DynamicContext) {
	refid := elem.SelectAttrValue("refid", "")
	fullRefid := b.Namespace + "." + refid
	var fragmentElem *etree.Element
	var ok bool

	// 首先检查是否包含完整的命名空间
	if fragmentElem, ok = GlobalSqlFragments[refid]; ok {
		GlobalSqlFragmentUsage[fullRefid]++
		b.parseDynamicTags(fragmentElem, context)
		return
	}

	// 如果没有找到，尝试使用当前命名空间
	if fragmentElem, ok = GlobalSqlFragments[fullRefid]; ok {
		b.parseDynamicTags(fragmentElem, context)
		GlobalSqlFragmentUsage[fullRefid]++
		return
	}

	// 如果仍然没有找到，添加到未知片段列表
	if refid != "" {
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
