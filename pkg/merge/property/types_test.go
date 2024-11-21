package property

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPropertyEntryBasic(t *testing.T) {
	e := PropertyEntry{Value: "a"}
	assert.Equal(t, "a", e.StringValue())
	assert.Equal(t, "", e.RawReferenceValue())

	e = PropertyEntry{Value: "${a.b}"}
	assert.Equal(t, "${a.b}", e.StringValue())
	assert.Equal(t, "${a.b}", e.RawReferenceValue())

	e = PropertyEntry{Value: true}
	assert.Equal(t, "", e.StringValue())
	assert.Equal(t, "", e.RawReferenceValue())

	e = PropertyEntry{Value: 1.8}
	assert.Equal(t, "", e.StringValue())
	assert.Equal(t, "", e.RawReferenceValue())
}
