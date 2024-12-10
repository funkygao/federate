package property

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreateAnnotationReferenceRegex(t *testing.T) {
	tests := []struct {
		key         string
		annotation  string
		shouldMatch bool
	}{
		{
			key:         "my.property",
			annotation:  `@Value("${my.property}")`,
			shouldMatch: true,
		},
		{
			key:         "my.property",
			annotation:  `@Value("#{'${my.property}'.split(',')}")`,
			shouldMatch: true,
		},
		{
			key:         "my.property",
			annotation:  `@ConditionalOnProperty(name = "my.property")`,
			shouldMatch: true,
		},
		{
			key:         "my.property",
			annotation:  `@ConditionalOnProperty(name = "my.property", prefix ="foo")`,
			shouldMatch: true,
		},
		{
			key:         "my.property",
			annotation:  `@ConfigurationProperties(prefix = "my.property")`,
			shouldMatch: false, // not in []annotations
		},
		{
			key:         "my.property",
			annotation:  `@Value("${other.property}")`,
			shouldMatch: false,
		},
		{
			key:         "my.property",
			annotation:  `@CustomAnnotation("${my.property}")`,
			shouldMatch: false,
		},
		{
			key:         "server.servlet.context-path",
			annotation:  `@RequestMapping("/api")`,
			shouldMatch: false,
		},
		{
			key:         "endpoint",
			annotation:  `@GetMapping(value = "/${endpoint}")`,
			shouldMatch: true,
		},
		{
			key:         "list.values",
			annotation:  `@Value("#{'${list.values}'.split(',')}")`,
			shouldMatch: true,
		},
		{
			key:         "my.property",
			annotation:  `@Value("${my.property:default}")`,
			shouldMatch: true,
		},
	}

	for _, tt := range tests {
		regex := P.createAnnotationReferenceRegex(tt.key)
		assert.NotNil(t, regex, "Regex should not be nil for key: %s", tt.key)

		matches := regex.FindStringSubmatch(tt.annotation)
		if tt.shouldMatch {
			assert.True(t, len(matches) > 0, "Expected annotation to match for key '%s': %s", tt.key, tt.annotation)
		} else {
			assert.False(t, len(matches) > 0, "Expected annotation NOT to match for key '%s': %s", tt.key, tt.annotation)
		}
	}
}
