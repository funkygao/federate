package merge

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMethodResourcePattern(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name: "Simple @Resource on method",
			input: `@Resource
    public void setService(SomeService service) {`,
			expected: true,
		},
		{
			name: "Resource with name on method",
			input: `@Resource(name = "customName")
    public void setService(SomeService service) {`,
			expected: true,
		},
		{
			name: "Resource with whitespace",
			input: `  @Resource
    public void setService(SomeService service) {`,
			expected: true,
		},
		{
			name: "Resource on field (should not match)",
			input: `@Resource
    private SomeService service;`,
			expected: false,
		},
		{
			name: "Autowired on method (should not match)",
			input: `@Autowired
    public void setService(SomeService service) {`,
			expected: false,
		},
		{
			name: "Resource on non-setter method (should not match)",
			input: `@Resource
    public SomeService getService() {`,
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := P.methodResourcePattern.MatchString(tc.input)
			assert.Equal(t, tc.expected, result, "For input: %s", tc.input)
		})
	}
}
