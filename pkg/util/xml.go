package util

import (
	"github.com/beevik/etree"
)

func UpdateXmlElement(elem *etree.Element, attr string, newValue string) (updated bool) {
	if elem.SelectAttr(attr) != nil {
		elem.RemoveAttr(attr)
		elem.CreateAttr(attr, newValue)
		updated = true
	}
	return

}
