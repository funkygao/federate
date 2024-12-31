package mybatis

import (
	"github.com/beevik/etree"
)

func (mp *MyBatisProcessor) ExtractSQLFragments(root *etree.Element) {
	for _, elem := range root.ChildElements() {
		if elem.Tag == "sql" {
			if id := elem.SelectAttrValue("id", ""); id != "" {
				mp.fragments[id] = mp.extractRawSQL(elem)
			}
		}
	}
}
