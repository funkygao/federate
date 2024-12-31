package mybatis

import (
	"strings"

	"github.com/beevik/etree"
)

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
