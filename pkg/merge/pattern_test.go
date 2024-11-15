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

func TestPackageRegex(t *testing.T) {
	assert := assert.New(t)

	validPackages := []string{
		"package com.example;",
		"package com.example.subpackage;",
		"package com.example",
		"  package com.example;  ",
		"package com.example.subpackage",
	}

	invalidPackages := []string{
		"packag com.example;",
		"package ",
		"package;",
		"import com.example;",
		"public class Example {",
	}

	for _, pkg := range validPackages {
		assert.True(P.packageRegex.MatchString(pkg), "Should be a valid package declaration: %s", pkg)
	}

	for _, pkg := range invalidPackages {
		assert.False(P.packageRegex.MatchString(pkg), "Should be an invalid package declaration: %s", pkg)
	}
}

func TestImportRegex(t *testing.T) {
	assert := assert.New(t)

	validImports := []string{
		"import java.util.List;",
		"import static java.lang.Math.PI;",
		"import java.util.*;",
		"  import java.io.File;  ",
		"import java.util.concurrent.atomic.AtomicInteger",
		"import static org.junit.Assert.*",
	}

	invalidImports := []string{
		"impor java.util.List;",
		"import ",
		"import;",
		"package com.example;",
		"public class Example {",
	}

	for _, imp := range validImports {
		assert.True(P.importRegex.MatchString(imp), "Should be a valid import statement: %s", imp)
	}

	for _, imp := range invalidImports {
		assert.False(P.importRegex.MatchString(imp), "Should be an invalid import statement: %s", imp)
	}
}
