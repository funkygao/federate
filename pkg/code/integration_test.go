package code

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestApplyBeanTransforms(t *testing.T) {
	tests := []struct {
		name           string
		content        string
		beanTransforms map[string]string
		expected       string
	}{
		{
			name: "Autowired annotation",
			content: `
                @Autowired("oldBean")
                private SomeType someField;
            `,
			beanTransforms: map[string]string{"oldBean": "newBean"},
			expected: `
                @Autowired("newBean")
                private SomeType someField;
            `,
		},
		{
			name: "Qualifier annotation",
			content: `
                @Autowired
                @Qualifier("oldBean")
                private SomeType someField;
            `,
			beanTransforms: map[string]string{"oldBean": "newBean"},
			expected: `
                @Autowired
                @Qualifier("newBean")
                private SomeType someField;
            `,
		},
		{
			name: "Resource annotation",
			content: `
                @Resource(name = "oldBean")
                private SomeType someField;
            `,
			beanTransforms: map[string]string{"oldBean": "newBean"},
			expected: `
                @Resource(name = "newBean")
                private SomeType someField;
            `,
		},
		{
			name: "Multiple transformations",
			content: `
                @Autowired("oldBean1")
                private SomeType field1;

                @Qualifier("oldBean2")
                @Autowired
                private SomeType field2;

                @Resource(name = "oldBean3")
                private SomeType field3;
            `,
			beanTransforms: map[string]string{
				"oldBean1": "newBean1",
				"oldBean2": "newBean2",
				"oldBean3": "newBean3",
			},
			expected: `
                @Autowired("newBean1")
                private SomeType field1;

                @Qualifier("newBean2")
                @Autowired
                private SomeType field2;

                @Resource(name = "newBean3")
                private SomeType field3;
            `,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jf := NewJavaFile("", nil, tt.content)
			result := jf.applyBeanTransformRule(tt.beanTransforms)
			assert.Equal(t, tt.expected, result, tt.name)
		})
	}
}

func TestScanBeanTypeCounts(t *testing.T) {
	testCases := []struct {
		name     string
		input    []string
		expected map[string][]string
	}{
		{
			name: "Field injections",
			input: []string{
				"@Resource",
				"private SomeService service1;",
				"@Autowired",
				"private SomeService service2;",
				"@Resource",
				"private OtherService otherService;",
			},
			expected: map[string][]string{
				"SomeService":  []string{"service1", "service2"},
				"OtherService": []string{"otherService"},
			},
		},
		{
			name: "Method injections",
			input: []string{
				"@Resource",
				"public void setService1(SomeService service) {",
				"}",
				"@Resource",
				"public void setService2(SomeService service) {",
				"}",
				"@Resource",
				"public void setOtherService(OtherService service) {",
				"}",
			},
			expected: map[string][]string{
				"SomeService":  []string{"service1", "service2"},
				"OtherService": []string{"otherService"},
			},
		},
		{
			name: "Mixed field and method injections",
			input: []string{
				"@Resource",
				"private SomeService service1;",
				"@Resource",
				"public void setService2(SomeService service) {",
				"}",
				"@Autowired",
				"private OtherService otherService;",
			},
			expected: map[string][]string{
				"SomeService":  []string{"service1", "service2"},
				"OtherService": []string{"otherService"},
			},
		},
		{
			name: "Generic types",
			input: []string{
				"@Resource",
				"private List<String> stringList;",
				"@Autowired",
				"private Map<String, Object> objectMap;",
			},
			expected: map[string][]string{},
		},
		{
			name: "No injections",
			input: []string{
				"public class TestClass {",
				"    private SomeService service;",
				"    public void doSomething() {}",
				"}",
			},
			expected: map[string][]string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jl := NewJavaLines(tc.input)
			result := jl.ScanInjectedBeans()
			assert.Equal(t, tc.expected, result, tc.name)
		})
	}
}

func TestGetBeanTypeFromMethodSignature(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Standard setter method",
			input:    "public void setService(SomeService service) {",
			expected: "SomeService",
		},
		{
			name:     "Setter method with generic type",
			input:    "public void setList(List<String> list) {",
			expected: "List<String>",
		},
		{
			name:     "Setter method with multiple parameters",
			input:    "public void setComplexService(SomeService service, int value) {",
			expected: "SomeService",
		},
		{
			name:     "Non-setter method",
			input:    "public SomeService getService() {",
			expected: "",
		},
		{
			name:     "Method with no parameters",
			input:    "public void doSomething() {",
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jl := NewJavaLines(strings.Split(tc.input, "\n"))
			result := jl.GetBeanTypeFromMethodSignature(tc.input)
			assert.Equal(t, tc.expected, result, "For input: %s", tc.input)
		})
	}
}

func TestGetQualifierNameFromMethod(t *testing.T) {
	testCases := []struct {
		name         string
		resourceLine string
		methodLine   string
		expected     string
	}{
		{
			name:         "Resource with explicit name",
			resourceLine: "@Resource(name = \"customName\")",
			methodLine:   "public void setService(SomeService service) {",
			expected:     "customName",
		},
		{
			name:         "Resource without name, standard setter",
			resourceLine: "@Resource",
			methodLine:   "public void setService(SomeService service) {",
			expected:     "service",
		},
		{
			name:         "Resource without name, capitalized setter",
			resourceLine: "@Resource",
			methodLine:   "public void setMyService(SomeService service) {",
			expected:     "myService",
		},
		{
			name:         "Resource without name, non-standard setter",
			resourceLine: "@Resource",
			methodLine:   "public void setupService(SomeService service) {",
			expected:     "",
		},
		{
			name:         "No Resource annotation",
			resourceLine: "",
			methodLine:   "public void setService(SomeService service) {",
			expected:     "service",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jl := NewJavaLines(nil)
			result := jl.GetQualifierNameFromMethod(tc.resourceLine, tc.methodLine)
			assert.Equal(t, tc.expected, result, "For resource line: %s, method line: %s", tc.resourceLine, tc.methodLine)
		})
	}
}

func TestSpringBeanInjectionManager_shouldKeepResource(t *testing.T) {
	tests := []struct {
		name           string
		beans          map[string][]string
		beanType       string
		fieldName      string
		expectedResult bool
	}{
		{
			name: "Should keep resource - matching pattern with Impl",
			beans: map[string][]string{
				"FooService": {"fooService", "fooServiceImpl"},
			},
			beanType:       "FooService",
			fieldName:      "fooService",
			expectedResult: true,
		},
		{
			name: "Should keep resource - matching pattern with Impl (reverse check)",
			beans: map[string][]string{
				"FooService": {"fooService", "fooServiceImpl"},
			},
			beanType:       "FooService",
			fieldName:      "fooServiceImpl",
			expectedResult: true,
		},
		{
			name: "Should not keep resource - no matching Impl pattern",
			beans: map[string][]string{
				"BarService": {"barService", "anotherBarService"},
			},
			beanType:       "BarService",
			fieldName:      "barService",
			expectedResult: false,
		},
		{
			name: "Should not keep resource - single instance",
			beans: map[string][]string{
				"FooService": {"fooService"},
			},
			beanType:       "FooService",
			fieldName:      "fooService",
			expectedResult: false,
		},
		{
			name: "Should not keep resource - multiple instances but no Impl pattern",
			beans: map[string][]string{
				"FooService": {"fooService", "anotherFooService", "yetAnotherFooService"},
			},
			beanType:       "FooService",
			fieldName:      "fooService",
			expectedResult: false,
		},
		{
			name: "Should not keep resource - bean type not in map",
			beans: map[string][]string{
				"BarService": {"barService"},
			},
			beanType:       "FooService",
			fieldName:      "fooService",
			expectedResult: false,
		},
		{
			name: "Should keep resource - matching pattern with and without Impl",
			beans: map[string][]string{
				"ChangeOrderDetailRepository": {"changeOrderDetailRepository", "changeOrderDetailRepositoryImpl"},
			},
			beanType:       "ChangeOrderDetailRepository",
			fieldName:      "changeOrderDetailRepository",
			expectedResult: true,
		},
		{
			name: "Should keep resource - matching pattern with and without Impl (reverse check)",
			beans: map[string][]string{
				"ChangeOrderDetailRepository": {"changeOrderDetailRepository", "changeOrderDetailRepositoryImpl"},
			},
			beanType:       "ChangeOrderDetailRepository",
			fieldName:      "changeOrderDetailRepositoryImpl",
			expectedResult: true,
		},
	}

	jf := NewJavaFile("", nil, "")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := jf.ShouldKeepResource(tt.beans, tt.beanType, tt.fieldName)
			assert.Equal(t, tt.expectedResult, result, tt.name)
		})
	}
}
