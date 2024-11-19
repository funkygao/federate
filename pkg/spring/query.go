package spring

import (
	"github.com/beevik/etree"
)

type Query struct {
	queryString string
	attributes  []string
	tags        map[string][]string
	predicate   func(*etree.Element) bool
}

func NewQuery(queryString string, attributes []string, tags map[string][]string, predicate func(*etree.Element) bool) Query {
	return Query{
		queryString: queryString,
		attributes:  attributes,
		tags:        tags,
		predicate:   predicate,
	}
}

// 预定义查询
var (
	beanFullTags = map[string]struct{}{
		"bean":               {},
		"util:map":           {},
		"util:list":          {},
		"laf-config:manager": {},
		"jmq:producer":       {},
		"jmq:consumer":       {},
		"jmq:transport":      {},
		"jsf:consumer":       {},
		"jsf:consumerGroup":  {},
		"jsf:provider":       {},
		"jsf:filter":         {},
		"jsf:server":         {},
		"jsf:registry":       {},
		"dubbo:reference":    {},
		"dubbo:service":      {},
	}

	QueryBeanID = func(id string) Query {
		return NewQuery(
			id,
			[]string{"id", "name"},
			nil,
			func(elem *etree.Element) bool {
				fullTag := elem.Tag
				if elem.Space != "" {
					fullTag = elem.Space + ":" + elem.Tag
				}
				_, present := beanFullTags[fullTag]
				return present
			},
		)
	}

	QueryRpcAlias = func() Query {
		return NewQuery(
			"",
			[]string{"alias", "group"},
			nil,
			func(elem *etree.Element) bool {
				return elem.Tag == "provider" || elem.Tag == "service"
			},
		)
	}

	QueryRef = func() Query {
		return NewQuery(
			"",
			[]string{"ref", "value-ref", "bean", "properties-ref"},
			map[string][]string{"ref": {"bean"}},
			nil,
		)
	}
)
