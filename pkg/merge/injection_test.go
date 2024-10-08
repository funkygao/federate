package merge

import (
	"testing"

	"federate/pkg/util"
	"github.com/stretchr/testify/assert"
)

func TestReplaceResourceWithAutowiredModifiedCases(t *testing.T) {
	manager := NewSpringBeanInjectionManager()

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "Remove Resource import but keep following empty line",
			input: `
package com.example;

import javax.annotation.Resource;

import org.springframework.stereotype.Component;

@Component
public class TestClass {
    @Resource
    private SomeService service;
}`,
			expected: `
package com.example;

import javax.annotation.Resource;

import org.springframework.stereotype.Component;

import org.springframework.beans.factory.annotation.Autowired;

@Component
public class TestClass {
    @Autowired
    private SomeService service;
}`,
		},

		{
			name: "Replace @Resource with @Autowired and add import",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource
    private SomeService service;
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Autowired
    private SomeService service;
}`,
		},

		{
			name: "Replace @Resource with @Autowired when wildcard import exists",
			input: `
package com.example;

import javax.annotation.*;

public class TestClass {
    @Resource
    private SomeService service;
}`,
			expected: `
package com.example;

import javax.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Autowired
    private SomeService service;
}`,
		},

		{
			name: "Keep wildcard import when other annotations are used",
			input: `
package com.example;

import javax.annotation.*;

public class TestClass {
    @Resource
    private SomeService service;

    @PostConstruct
    public void init() {}
}`,
			expected: `
package com.example;

import javax.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Autowired
    private SomeService service;

    @PostConstruct
    public void init() {}
}`,
		},

		{
			name: "Don't add @Autowired import if already present",
			input: `
package com.example;

import org.springframework.beans.factory.annotation.Autowired;
import javax.annotation.Resource;

public class TestClass {
    @Resource
    private SomeService service;
}`,
			expected: `
package com.example;

import org.springframework.beans.factory.annotation.Autowired;
import javax.annotation.Resource;

public class TestClass {
    @Autowired
    private SomeService service;
}`,
		},

		{
			name: "Replace multiple @Resource annotations",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource
    private SomeService service1;

    @Resource
    private OtherService service2;
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Autowired
    private SomeService service1;

    @Autowired
    private OtherService service2;
}`,
		},

		{
			name: "Replace @Resource with name parameter",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource(name = "jmq4.producer")
    private SomeService service;
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    @Autowired
    @Qualifier("jmq4.producer")
    private SomeService service;
}`,
		},

		{
			name: "Replace multiple @Resource annotations with and without name",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource(name = "service1")
    private SomeService service1;

    @Resource
    private OtherService service2;

    @Resource(name = "service3")
    private AnotherService service3;
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    @Autowired
    @Qualifier("service1")
    private SomeService service1;

    @Autowired
    private OtherService service2;

    @Autowired
    @Qualifier("service3")
    private AnotherService service3;
}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := manager.replaceResourceWithAutowired(tc.input)
			assert.Equal(t, util.RemoveEmptyLines(tc.expected), util.RemoveEmptyLines(result))
		})
	}
}

func TestReplaceResourceWithAutowiredNotChanged(t *testing.T) {
	manager := NewSpringBeanInjectionManager()

	testCases := []struct {
		name  string
		input string
	}{
		{
			name: "No changes when @Resource is not present",
			input: `
package com.example;

import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Autowired
    private SomeService service1;

    @Autowired
    private OtherService service2;
}`,
		},

		{
			name: "Keep @Resource for Map injection",
			input: `
package com.example;

import javax.annotation.Resource;
import org.springframework.stereotype.Component;

@Component
@Order(60)
public class LocationValidationRuleInitializer implements LocationValidationInitializer {
    @Resource
    private Map<String, String> mappingConstraintValidatorMap;
}`,
		},

		{
			name: "Keep @Resource for HashMap injection",
			input: `
package com.example;

import javax.annotation.Resource;
import org.springframework.stereotype.Component;

@Component
@Order(60)
public class LocationValidationRuleInitializer implements LocationValidationInitializer {
    @Resource
    private HashMap<String, String> mappingConstraintValidatorMap;
}`,
		},

		{
			name: "Keep @Resource for List injection",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource
    private List<String> stringList;
}`,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := manager.replaceResourceWithAutowired(tc.input)
			assert.Equal(t, util.RemoveEmptyLines(tc.input), util.RemoveEmptyLines(result))
		})
	}
}

func TestProcessCodeLines(t *testing.T) {
	manager := NewSpringBeanInjectionManager()

	testCases := []struct {
		name              string
		input             []string
		expectedOutput    []string
		expectedAutowired bool
		expectedQualifier bool
	}{
		{
			name: "Simple @Resource to @Autowired",
			input: []string{
				"@Resource",
				"private SomeService service;",
			},
			expectedOutput: []string{
				"@Autowired",
				"private SomeService service;",
			},
			expectedAutowired: true,
			expectedQualifier: false,
		},
		{
			name: "@Resource with name to @Autowired and @Qualifier",
			input: []string{
				"@Resource(name = \"specificName\")",
				"private SomeService service;",
			},
			expectedOutput: []string{
				"@Autowired",
				"@Qualifier(\"specificName\")",
				"private SomeService service;",
			},
			expectedAutowired: true,
			expectedQualifier: true,
		},
		{
			name: "Keep @Resource for Map",
			input: []string{
				"@Resource",
				"private Map<String, String> map;",
			},
			expectedOutput: []string{
				"@Resource",
				"private Map<String, String> map;",
			},
			expectedAutowired: false,
			expectedQualifier: false,
		},
		{
			name: "Keep @Resource for List",
			input: []string{
				"@Resource",
				"private List<String> list;",
			},
			expectedOutput: []string{
				"@Resource",
				"private List<String> list;",
			},
			expectedAutowired: false,
			expectedQualifier: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output, needAutowired, needQualifier := manager.processCodeLines(tc.input)
			assert.Equal(t, tc.expectedOutput, output)
			assert.Equal(t, tc.expectedAutowired, needAutowired)
			assert.Equal(t, tc.expectedQualifier, needQualifier)
		})
	}
}

func TestProcessImports(t *testing.T) {
	manager := NewSpringBeanInjectionManager()

	testCases := []struct {
		name            string
		imports         []string
		needAutowired   bool
		needQualifier   bool
		expectedImports []string
	}{
		{
			name: "Keep existing Qualifier import",
			imports: []string{
				"import org.springframework.beans.factory.annotation.Qualifier;",
				"import some.other.package;",
			},
			needAutowired: true,
			needQualifier: true,
			expectedImports: []string{
				"import org.springframework.beans.factory.annotation.Qualifier;",
				"import some.other.package;",
				"import org.springframework.beans.factory.annotation.Autowired;",
			},
		},
		{
			name: "Add both Autowired and Qualifier imports",
			imports: []string{
				"import some.other.package;",
			},
			needAutowired: true,
			needQualifier: true,
			expectedImports: []string{
				"import some.other.package;",
				"import org.springframework.beans.factory.annotation.Autowired;",
				"import org.springframework.beans.factory.annotation.Qualifier;",
			},
		},
		{
			name: "Don't add unnecessary imports",
			imports: []string{
				"import org.springframework.beans.factory.annotation.Autowired;",
				"import org.springframework.beans.factory.annotation.Qualifier;",
			},
			needAutowired: false,
			needQualifier: false,
			expectedImports: []string{
				"import org.springframework.beans.factory.annotation.Autowired;",
				"import org.springframework.beans.factory.annotation.Qualifier;",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := manager.processImports(tc.imports, tc.needAutowired, tc.needQualifier)
			assert.Equal(t, tc.expectedImports, result)
		})
	}
}
