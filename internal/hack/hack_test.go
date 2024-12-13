package hack

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestB2s(t *testing.T) {
	b := []byte("hello world")
	assert.Equal(t, B2s(b), "hello world")
}

func TestS2b(t *testing.T) {
	s := "hello world"
	assert.Equal(t, []byte(s), S2b(s))
}
