package primitive

type StringSet struct {
	items map[string]struct{}
}

func NewStringSet() *StringSet {
	return &StringSet{
		items: make(map[string]struct{}),
	}
}

func (s *StringSet) Add(item string) {
	s.items[item] = struct{}{}
}

func (s *StringSet) Contains(item string) bool {
	_, exists := s.items[item]
	return exists
}

func (s *StringSet) Items() map[string]struct{} {
	return s.items
}

func (s *StringSet) Remove(item string) {
	delete(s.items, item)
}

func (s *StringSet) Clear() {
	s.items = make(map[string]struct{})
}

func (s *StringSet) Size() int {
	return len(s.items)
}
