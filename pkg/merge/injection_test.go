package merge

import (
	"strings"
	"testing"

	"federate/pkg/code"
	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/stretchr/testify/assert"
)

func TestReplaceResourceWithAutowiredModifiedCases(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	testCases := []struct {
		name     string
		input    string
		expected string
		dirty    bool
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
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
			dirty: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jf := code.NewJavaFile("", nil, []byte(tc.input))
			result, dirty := manager.reconcileInjectionAnnotations(jf)
			assert.Equal(t, util.RemoveEmptyLines(tc.expected), util.RemoveEmptyLines(result))
			assert.Equal(t, dirty, tc.dirty, tc.name)
		})
	}
}

func TestReplaceResourceWithAutowiredNotChanged(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

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
			jf := code.NewJavaFile("", nil, []byte(tc.input))
			result, _ := manager.reconcileInjectionAnnotations(jf)
			assert.Equal(t, util.RemoveEmptyLines(tc.input), util.RemoveEmptyLines(result))
		})
	}
}

func TestProcessCodeLines(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

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
			output, needAutowired, needQualifier := manager.transformInjectionAnnotations(nil, tc.input)
			assert.Equal(t, tc.expectedOutput, output)
			assert.Equal(t, tc.expectedAutowired, needAutowired)
			assert.Equal(t, tc.expectedQualifier, needQualifier)
		})
	}
}

func TestTransformImportIfNec(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

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
			result := manager.transformImportIfNec(tc.imports, tc.needAutowired, tc.needQualifier)
			assert.Equal(t, tc.expectedImports, result)
		})
	}
}

func TestReplaceResourceWithAutowiredForMultipleInstances(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	input := `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource
    private Cluster c1;

    @Resource
    private Cluster c2;

    @Resource(name = "cluster3")
    private Cluster c3;

    @Resource
    private AnotherType a1;

    @Resource
    private AnotherType a2;
}
`

	expected := `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    @Autowired
    @Qualifier("c1")
    private Cluster c1;

    @Autowired
    @Qualifier("c2")
    private Cluster c2;

    @Autowired
    @Qualifier("cluster3")
    private Cluster c3;

    @Autowired
    @Qualifier("a1")
    private AnotherType a1;

    @Autowired
    @Qualifier("a2")
    private AnotherType a2;
}
`

	jf := code.NewJavaFile("", nil, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, util.RemoveEmptyLines(expected), util.RemoveEmptyLines(result))
}

func TestParseFieldDeclaration(t *testing.T) {
	testCases := []struct {
		name          string
		input         string
		expectedType  string
		expectedField string
	}{
		{
			name:          "Standard private field",
			input:         "private Cluster c1;",
			expectedType:  "Cluster",
			expectedField: "c1",
		},
		{
			name:          "Field without access modifier",
			input:         "Cluster c2;",
			expectedType:  "Cluster",
			expectedField: "c2",
		},
		{
			name:          "Protected field",
			input:         "protected AnotherType a1;",
			expectedType:  "AnotherType",
			expectedField: "a1",
		},
		{
			name:          "Public field",
			input:         "public Map<String, Object> map;",
			expectedType:  "Map<String,Object>",
			expectedField: "map",
		},
		{
			name:          "Field with generic type",
			input:         "private List<String> stringList;",
			expectedType:  "List<String>",
			expectedField: "stringList",
		},
		{
			name:          "Field with complex generic type",
			input:         "private Map<String, List<Integer>> complexMap;",
			expectedType:  "Map<String,List<Integer>>",
			expectedField: "complexMap",
		},
		{
			name:          "Field with nested generic type",
			input:         "private Pair<List<String>, Map<Integer, Set<Double>>> nestedGeneric;",
			expectedType:  "Pair<List<String>,Map<Integer,Set<Double>>>",
			expectedField: "nestedGeneric",
		},
		{
			name:          "Field with initialization",
			input:         "private int count = 0;",
			expectedType:  "int",
			expectedField: "count",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jl := code.NewJavaLines(strings.Split(tc.input, "\n"))
			beanType, fieldName := jl.ParseFieldDeclaration(tc.input)
			assert.Equal(t, tc.expectedType, beanType, "Bean type mismatch for input: %s", tc.input)
			assert.Equal(t, tc.expectedField, fieldName, "Field name mismatch for input: %s", tc.input)
		})
	}
}

func TestReplaceResourceWithAutowiredComments(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	input := `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    // @Resource
    private SomeService service1;

    /*
     * @Resource
     */
    private OtherService service2;

    /**
     * @Resource
     */
    private AnotherService service3;

    @Resource
    private RealService service4;
}
`

	expected := `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    // @Resource
    private SomeService service1;

    /*
     * @Resource
     */
    private OtherService service2;

    /**
     * @Resource
     */
    private AnotherService service3;

    @Autowired
    private RealService service4;
}
`

	jf := code.NewJavaFile("", nil, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, util.RemoveEmptyLines(expected), util.RemoveEmptyLines(result))
}

func TestReplaceResourceWithAutowiredSpecialCases(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	input := `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    @Resource
    private SomeService service1;
    @Autowired
    private SomeService service2;
    @Autowired(required = false)
    private SomeService service3;
    @Resource(name = "specificName")
    private SomeService service4;
}
`

	expected := `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    @Autowired
    @Qualifier("service1")
    private SomeService service1;
    @Autowired
    @Qualifier("service2")
    private SomeService service2;
    @Autowired(required = false)
    @Qualifier("service3")
    private SomeService service3;
    @Autowired
    @Qualifier("specificName")
    private SomeService service4;
}
`
	jf := code.NewJavaFile("", nil, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, util.RemoveEmptyLines(expected), util.RemoveEmptyLines(result))
}

func TestReplaceResourceWithAutowiredImports(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	input := `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    @Resource
    private SomeService service1;

    @Resource
    private SomeService service2;
}
`

	expected := `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    @Autowired
    @Qualifier("service1")
    private SomeService service1;

    @Autowired
    @Qualifier("service2")
    private SomeService service2;
}
`

	jf := code.NewJavaFile("", nil, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, util.RemoveEmptyLines(expected), util.RemoveEmptyLines(result))
}

func TestReplaceResourceWithAutowiredOnMethods(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "Replace @Resource on setter method",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    private SomeService service;

    @Resource
    public void setService(SomeService service) {
        this.service = service;
    }
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

public class TestClass {
    private SomeService service;

    @Autowired
    public void setService(SomeService service) {
        this.service = service;
    }
}`,
		},

		{
			name: "Replace @Resource with name on setter method",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    private SomeService service;

    @Resource(name = "customServiceName")
    public void setService(SomeService service) {
        this.service = service;
    }
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    private SomeService service;

    @Autowired
    @Qualifier("customServiceName")
    public void setService(SomeService service) {
        this.service = service;
    }
}`,
		},

		{
			name: "Replace multiple @Resource annotations on methods",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    private SomeService service1;
    private OtherService service2;

    @Resource
    public void setService1(SomeService service) {
        this.service1 = service;
    }

    @Resource(name = "customName")
    public void setService2(OtherService service) {
        this.service2 = service;
    }
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    private SomeService service1;
    private OtherService service2;

    @Autowired
    public void setService1(SomeService service) {
        this.service1 = service;
    }

    @Autowired
    @Qualifier("customName")
    public void setService2(OtherService service) {
        this.service2 = service;
    }
}`,
		},

		{
			name: "Replace @Resource on setter method with multiple beans of the same type",
			input: `
package com.example;

import javax.annotation.Resource;

public class TestClass {
    private SomeService service1;
    private SomeService service2;

    @Resource
    public void setService1(SomeService service) {
        this.service1 = service;
    }
    @Resource
    public void setService2(SomeService service) {
        this.service2 = service;
    }
}`,
			expected: `
package com.example;

import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {
    private SomeService service1;
    private SomeService service2;

    @Autowired
    @Qualifier("service1")
    public void setService1(SomeService service) {
        this.service1 = service;
    }
    @Autowired
    @Qualifier("service2")
    public void setService2(SomeService service) {
        this.service2 = service;
    }
}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			jf := code.NewJavaFile("", nil, []byte(tc.input))
			result, _ := manager.reconcileInjectionAnnotations(jf)
			assert.Equal(t, util.RemoveEmptyLines(tc.expected), util.RemoveEmptyLines(result), tc.name)
		})
	}
}

func TestSpringBeanInjectionManager_transformInjectionAnnotations_ChangeOrderDetailRepository(t *testing.T) {
	manager := NewSpringBeanInjectionManager(nil)

	content := `
public class ChangeOrderApproveInnerAppServiceImpl implements ChangeOrderApproveInnerAppService {
    @Resource
    private ChangeOrderRepository changeOrderRepository;
    @Resource
    private ChangeOrderDetailRepository changeOrderDetailRepositoryImpl;
    @Resource
    private ExtChangeReasonService extChangeReasonServiceImpl;
    @Resource
    private ChangeOrderDetailRepository changeOrderDetailRepository;
}`

	jf := code.NewJavaFile("", nil, []byte(content))
	processedLines, needAutowired, needQualifier := manager.transformInjectionAnnotations(jf, jf.RawLines())

	// 验证结果
	assert.True(t, needAutowired)
	assert.False(t, needQualifier)

	expectedOutput := `
public class ChangeOrderApproveInnerAppServiceImpl implements ChangeOrderApproveInnerAppService {
    @Autowired
    private ChangeOrderRepository changeOrderRepository;
    @Resource
    private ChangeOrderDetailRepository changeOrderDetailRepositoryImpl;
    @Autowired
    private ExtChangeReasonService extChangeReasonServiceImpl;
    @Resource
    private ChangeOrderDetailRepository changeOrderDetailRepository;
}`

	assert.Equal(t, expectedOutput, strings.Join(processedLines, "\n"))
}

func TestNoExtraLinesBetweenImportsAndClass(t *testing.T) {

	input := `
// bingo
package com.example;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Method;
import java.util.*;

// hello world
import javax.annotation.Resource;

// import fake
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.ArrayUtils;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

@Component
public class TestClass {
    @Resource
    private SomeService service;
}
`

	expected := `
// bingo
package com.example;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Method;
import java.util.*;

// hello world
import javax.annotation.Resource;

// import fake
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.ArrayUtils;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.springframework.beans.factory.annotation.Autowired;

@Component
public class TestClass {
    @Autowired
    private SomeService service;
}
`

	manager := NewSpringBeanInjectionManager(nil)
	jf := code.NewJavaFile("", nil, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, expected, result, "Should not add extra lines between imports and class declaration")
}

func TestAutowiredAddQualifier(t *testing.T) {
	input := `
package foo;
import com.github.kv.cli.Cluster;
import org.springframework.beans.factory.annotation.Autowired;

@Component
public class KvTool {
    /**
     * 通用连接池
     */
    @Autowired(required = false)
    private Cluster commonClient;

    @Autowired(required = false)
    private Cluster myClient;
}
`

	expected := `
package foo;
import com.github.kv.cli.Cluster;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

@Component
public class KvTool {
    /**
     * 通用连接池
     */
    @Autowired(required = false)
    @Qualifier("commonClient")
    private Cluster commonClient;

    @Autowired(required = false)
    @Qualifier("myClient")
    private Cluster myClient;
}
`
	manager := NewSpringBeanInjectionManager(nil)
	jf := code.NewJavaFile("", nil, []byte(input))
	assert.True(t, jf.HasInjectionAnnotation(), "jf.HasInjectionAnnotation()")
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, expected, result)

}

func TestTransformAutowiredExclude(t *testing.T) {
	input := `
package foo;
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

@Setter
@Service
public class FooService implements Foo {
    @Resource
    private CountTaskMasterExternalService masterExternalService;
    @Resource
    private CountTaskMasterExternalService countTaskMasterExternalService;
    @Resource
    private Egg egg;
}
`

	expected := `
package foo;
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;

@Setter
@Service
public class FooService implements Foo {
    @Resource
    private CountTaskMasterExternalService masterExternalService;
    @Resource
    private CountTaskMasterExternalService countTaskMasterExternalService;
    @Autowired
    private Egg egg;
}
`
	manager := NewSpringBeanInjectionManager(nil)
	jf := code.NewJavaFile("FooService.java", &manifest.ComponentInfo{
		Name: "inv",
		Transform: manifest.TransformSpec{
			Autowired: manifest.AutowiredSpec{
				Excludes: []string{"CountTaskMasterExternalService"},
			},
		},
	}, []byte(input))
	result, _ := manager.reconcileInjectionAnnotations(jf)
	assert.Equal(t, expected, result)

}
