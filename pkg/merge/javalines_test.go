package merge

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSeparateSections(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		expectedHeader string
	}{
		{
			name: "Basic separation",
			input: `
package com.example;

import java.util.List;
import java.util.Map;

public class Test {
}`,
			expectedHeader: `package com.example;

import java.util.List;
import java.util.Map;`,
		},
		{
			name: "With comments containing package and import",
			input: `
// This comment contains package com.fake;
package com.example;

/* Multi-line comment
   import java.fake.Class;
   package com.another.fake; */
import java.util.List;
// Another comment with import java.util.FakeClass;
import java.util.Map;

public class Test {
    // import java.util.ArrayList;
}`,
			expectedHeader: `// This comment contains package com.fake;
package com.example;

/* Multi-line comment
   import java.fake.Class;
   package com.another.fake; */
import java.util.List;
// Another comment with import java.util.FakeClass;
import java.util.Map;`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lines := strings.Split(strings.TrimSpace(tt.input), "\n")
			jl := newJavaLines(lines)
			jl.SeparateSections()

			assert.Equal(t, tt.expectedHeader, strings.Join(jl.HeadLines(), "\n"))
		})
	}
}
