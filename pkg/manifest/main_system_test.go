package manifest

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExcludeJavaFile(t *testing.T) {
	testCases := []struct {
		input          string
		excludedTypes  []string
		expectedOutput bool
	}{
		{
			"Foo.java",
			[]string{"com.goog.Foo", "com.x.Bar"},
			true,
		},
		{
			"a/b/c/Foo.java",
			[]string{"com.goog.Foo", "com.x.Bar"},
			true,
		},
		{
			"a/b/c/Order.java",
			[]string{"com.goog.Foo", "com.x.Bar"},
			false,
		},
	}

	for _, tc := range testCases {
		m := MainClassSpec{
			ComponentScan: ComponentScanSpec{
				ExcludedTypes: tc.excludedTypes,
			},
		}
		assert.Equal(t, tc.expectedOutput, m.ExcludeJavaFile(tc.input))
	}

}
