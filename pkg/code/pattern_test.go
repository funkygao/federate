package code

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

func TestIsResourceAnnotatedLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected bool
	}{
		{"Simple @Resource", "@Resource", true},
		{"@Resource with parentheses", "@Resource()", true},
		{"@Resource with name", "@Resource(name = \"foo\")", true},
		{"Not @Resource", "public class Foo", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsResourceAnnotatedLine(tt.line)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsResourceAnnotatedWithNameLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected bool
	}{
		{"@Resource with name", "@Resource(name = \"foo\")", true},
		{"@Resource without name", "@Resource", false},
		{"@Resource with empty name", "@Resource(name = \"\")", true},
		{"Not @Resource", "public class Foo", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsResourceAnnotatedWithNameLine(tt.line)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetResourceAnnotationName(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected string
	}{
		{"@Resource with name", "@Resource(name = \"foo\")", "foo"},
		{"@Resource with empty name", "@Resource(name = \"\")", ""},
		{"@Resource without name", "@Resource", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetResourceAnnotationName(tt.line)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsMethodResourceAnnotatedLines(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		nextLine string
		expected bool
	}{
		{
			"Valid method resource",
			"@Resource",
			"public void setFoo(Foo foo) {",
			true,
		},
		{
			"Invalid method resource",
			"@Resource",
			"private void setFoo(Foo foo) {",
			false,
		},
		{
			"Not a method resource",
			"public class Foo {",
			"private int bar;",
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsMethodResourceAnnotatedLines(tt.line + "\n" + tt.nextLine)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsInjectionAnnotatedLineBasic(t *testing.T) {
	assert.False(t, IsInjectionAnnotatedLine(`public class`))
	assert.False(t, IsInjectionAnnotatedLine(`package foo;`))

	assert.True(t, IsInjectionAnnotatedLine(`@Autowired`))
	assert.True(t, IsInjectionAnnotatedLine(`@Autowired(required = false)`))
	assert.False(t, IsInjectionAnnotatedLine(`@Autowiredx(required = false)`), `@Autowiredx(required = false)`)
	assert.True(t, IsInjectionAnnotatedLine(`@Resource`))
	assert.True(t, IsInjectionAnnotatedLine(`@Resource(name ="foo")`))
	assert.False(t, IsInjectionAnnotatedLine(`@ResourceMapper(name ="foo")`), `@ResourceMapper(name ="foo")`)
}

func TestIsInjectionAnnotatedLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected bool
	}{
		{"@Resource", "@Resource", true},
		{"@Autowired", "@Autowired", true},
		{"@Resource with parentheses", "@Resource()", true},
		{"@Autowired with parentheses", "@Autowired()", true},
		{"Not injection", "public class Foo", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsInjectionAnnotatedLine(tt.line)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsCollectionTypeLine(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		expected bool
	}{
		{"Map type", "Map<String, Integer> map;", true},
		{"List type", "List<String> list;", true},
		{"HashMap type", "HashMap<String, Object> hashMap;", true},
		{"Not collection type", "String foo;", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsCollectionTypeLine(tt.line)
			assert.Equal(t, tt.expected, result)
		})
	}
}
