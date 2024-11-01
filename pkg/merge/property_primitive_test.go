package merge

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKey(t *testing.T) {
	k := Key("")
	assert.Equal(t, "stock.", k.NamespacePrefix("stock"))

	k = Key("mysql.uri")
	assert.Equal(t, "stock.mysql.uri", k.WithNamespace("stock"))
}
