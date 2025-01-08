package primitive

import "sort"

type StringSet struct {
	items map[string]struct{}

	useRaw   bool
	rawItems []string
}

func NewStringSet() *StringSet {
	return &StringSet{
		items:  make(map[string]struct{}),
		useRaw: false,
	}
}

func (s *StringSet) UseRaw() *StringSet {
	s.useRaw = true
	s.rawItems = make([]string, 0, 1<<8)
	return s
}

func (s *StringSet) Add(item string) {
	s.items[item] = struct{}{}
	if s.useRaw {
		s.rawItems = append(s.rawItems, item)
	}
}

func (s *StringSet) Contains(item string) bool {
	_, exists := s.items[item]
	return exists
}

func (s *StringSet) Items() map[string]struct{} {
	return s.items
}

func (s *StringSet) RawValues() []string {
	return s.rawItems
}

func (s *StringSet) Values() []string {
	r := make([]string, 0, s.Cardinality())
	for s := range s.items {
		r = append(r, s)
	}
	return r
}

func (s *StringSet) SortedValues() []string {
	values := s.Values()
	sort.Strings(values)
	return values
}

func (s *StringSet) Remove(item string) {
	delete(s.items, item)
}

func (s *StringSet) Clear() {
	s.items = make(map[string]struct{})
}

func (s *StringSet) Cardinality() int {
	return len(s.items)
}
