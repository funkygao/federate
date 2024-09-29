package merge

import (
	"testing"

	"federate/pkg/util"
	"github.com/stretchr/testify/assert"
)

func TestReplaceResourceWithAutowired(t *testing.T) {
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

import org.springframework.stereotype.Component;

import org.springframework.beans.factory.annotation.Autowired;

@Component
public class TestClass {
    @Autowired
    private SomeService service;
}`,
		},
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
			expected: `
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

import org.springframework.beans.factory.annotation.Autowired;
import javax.annotation.*;

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
