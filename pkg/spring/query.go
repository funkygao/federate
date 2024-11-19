package spring

import (
	"github.com/beevik/etree"
)

type Query struct {
	reserved SearchType

	query      string
	attributes []string
	tags       map[string][]string
	predicate  func(*etree.Element) bool
}

func (q *Query) SetQuery(query string) {
	q.query = query
}

func (q *Query) SetPredicate(p func(*etree.Element) bool) {
	q.predicate = p
}

var (
	QueryBeanID   = newReservedQuery(SearchByID, []string{"id", "name"}, nil)
	QueryRpcAlias = newReservedQuery(SearchByAlias, []string{"alias", "group"}, nil)
	QueryRef      = newReservedQuery(SearchByRef, []string{"ref", "value-ref", "bean", "properties-ref"},
		map[string][]string{
			"ref": {"bean"},
		})
)

func newReservedQuery(kind SearchType, attributes []string, tags map[string][]string) Query {
	query := NewQuery(attributes, tags)
	query.reserved = kind
	return query
}

func NewQuery(attributes []string, tags map[string][]string) Query {
	return Query{
		attributes: attributes,
		tags:       tags,
	}
}
