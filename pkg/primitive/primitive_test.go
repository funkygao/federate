package primitive

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValues(t *testing.T) {
	s := NewStringSet()
	s.Add("2")
	s.Add("1")
	s.Add("2")
	s.Add("5")
	assert.Equal(t, 3, s.Cardinality())

	values := s.Values()
	sort.Strings(values)
	assert.Equal(t, []string{"1", "2", "5"}, values)
}
