package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestContains(t *testing.T) {
	assert.True(t, Contains("a", []string{"a"}))
	assert.True(t, Contains("a", []string{"a", "b"}))
	assert.True(t, Contains("a", []string{"c", "a", "b"}))
	assert.False(t, Contains("m", []string{"c", "a", "b"}))
}
