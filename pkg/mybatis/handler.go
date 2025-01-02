package mybatis

import (
	"strings"

	"github.com/beevik/etree"
)

var (
	_ NodeHandler = (*TrimHandler)(nil)
	_ NodeHandler = (*WhereHandler)(nil)
	_ NodeHandler = (*ForEachHandler)(nil)
	_ NodeHandler = (*IfHandler)(nil) // when, if
	_ NodeHandler = (*SetHandler)(nil)
	_ NodeHandler = (*ChooseHandler)(nil)
	_ NodeHandler = (*OtherwiseHandler)(nil)
)

type NodeHandler interface {
	handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool
}

type DynamicContext struct {
	sql strings.Builder
}

// TrimHandler 处理 <trim> 标签
type TrimHandler struct{}

func (h *TrimHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	prefix := node.SelectAttrValue("prefix", "")
	suffixOverrides := node.SelectAttrValue("suffixOverrides", "")

	if strings.EqualFold(prefix, "set") {
		// 对于 SET 子句，我们直接简化处理
		context.sql.WriteString("SET foo=?")
		// 不需要处理 suffixOverrides，因为我们只写入一个 ?=?
	} else {
		// 对于其他情况，保持原有的处理逻辑
		childContent := &DynamicContext{}
		b.parseDynamicTags(node, childContent)
		trimmed := strings.TrimSpace(childContent.sql.String())

		if trimmed != "" {
			if prefix != "" {
				context.sql.WriteString(prefix)
				context.sql.WriteString(" ")
			}
			context.sql.WriteString(trimmed)
			// 处理 suffixOverrides
			if suffixOverrides != "" {
				for _, override := range strings.Split(suffixOverrides, "|") {
					trimmed = strings.TrimSuffix(trimmed, override)
				}
				context.sql.Reset()
				context.sql.WriteString(strings.TrimSpace(trimmed))
			}
		}
	}

	return true
}

// WhereHandler 处理 <where> 标签
type WhereHandler struct{}

func (h *WhereHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	childContent := &DynamicContext{}
	b.parseDynamicTags(node, childContent)
	trimmed := strings.TrimSpace(childContent.sql.String())

	if trimmed != "" {
		if strings.HasPrefix(strings.ToUpper(trimmed), "AND ") {
			trimmed = trimmed[4:]
		} else if strings.HasPrefix(strings.ToUpper(trimmed), "OR ") {
			trimmed = trimmed[3:]
		}
		context.sql.WriteString(" WHERE ")
		context.sql.WriteString(trimmed)
	}

	return true
}

// SetHandler 处理 <set> 标签
type SetHandler struct{}

func (h *SetHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	childContent := &DynamicContext{}
	b.parseDynamicTags(node, childContent)
	trimmed := strings.TrimSpace(childContent.sql.String())

	if trimmed != "" {
		if strings.HasSuffix(trimmed, ",") {
			trimmed = trimmed[:len(trimmed)-1]
		}
		context.sql.WriteString(" SET ")
		context.sql.WriteString(trimmed)
	}

	return true
}

// ForEachHandler 处理 <foreach> 标签
type ForEachHandler struct{}

func (h *ForEachHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	open := node.SelectAttrValue("open", "")
	close := node.SelectAttrValue("close", "")
	separator := node.SelectAttrValue("separator", "")

	context.sql.WriteString(open)
	context.sql.WriteString("/* FOREACH_START */ ")

	childContent := &DynamicContext{}
	b.parseDynamicTags(node, childContent)

	content := childContent.sql.String()

	items := strings.Split(content, separator)
	for i, item := range items {
		if i > 0 {
			context.sql.WriteString(separator)
		}
		context.sql.WriteString(strings.TrimSpace(item))
	}

	context.sql.WriteString(" /* FOREACH_END */")
	context.sql.WriteString(close)

	return true
}

// IfHandler 处理 <if> 标签
type IfHandler struct{}

func (h *IfHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	test := node.SelectAttrValue("test", "")
	// 这里应该有实际的条件评估逻辑
	// 现在我们假设所有条件都为真
	if test != "" {
		childContent := &DynamicContext{}
		b.parseDynamicTags(node, childContent)
		context.sql.WriteString(childContent.sql.String())
	}
	return true
}

// ChooseHandler 处理 <choose> 标签
type ChooseHandler struct{}

func (h *ChooseHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	for _, when := range node.SelectElements("when") {
		if b.nodeHandlers["if"].handle(b, when, context) {
			return true
		}
	}

	otherwise := node.SelectElement("otherwise")
	if otherwise != nil {
		b.parseDynamicTags(otherwise, context)
	}

	return true
}

// OtherwiseHandler 处理 <otherwise> 标签
type OtherwiseHandler struct{}

func (h *OtherwiseHandler) handle(b *XMLMapperBuilder, node *etree.Element, context *DynamicContext) bool {
	b.parseDynamicTags(node, context)
	return true
}

func newNodeHandlers() map[string]NodeHandler {
	return map[string]NodeHandler{
		"trim":      &TrimHandler{},
		"where":     &WhereHandler{},
		"set":       &SetHandler{},
		"foreach":   &ForEachHandler{},
		"if":        &IfHandler{},
		"choose":    &ChooseHandler{},
		"when":      &IfHandler{},
		"otherwise": &OtherwiseHandler{},
	}
}
