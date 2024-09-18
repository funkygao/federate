package federated

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPlaceholder(t *testing.T) {
	// This is a placeholder test to avoid "no test files" error.
}

func TestResourceBaseName(t *testing.T) {
	assert.Equal(t, "wms-stock/clover/spring-clover.xml", ResourceBaseName("generated/demo-project/src/main/resources/federated/wms-stock/clover/spring-clover.xml"))
	assert.Equal(t, "wms-stock/spring-clover.xml", ResourceBaseName("generated/demo-project/src/main/resources/federated/wms-stock/spring-clover.xml"))
}
