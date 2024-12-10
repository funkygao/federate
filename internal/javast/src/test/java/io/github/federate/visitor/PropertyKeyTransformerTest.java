package io.github.federate.visitor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.Expression;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PropertyKeyTransformerTest {
    private PropertyKeyTransformer transformer;
    private Map<String, String> keyMapping;

    @BeforeEach
    void setUp() {
        keyMapping = new HashMap<>();
        keyMapping.put("foo", "ns.foo");
        keyMapping.put("bar", "ns.bar");
        keyMapping.put("schedule.clover.webAlias", "wms-stock.schedule.clover.webAlias");
        keyMapping.put("sku.extendAttrs", "ns.sku.extendAttrs");
        transformer = new PropertyKeyTransformer(keyMapping);
    }

    @Test
    void testReplaceKeys() {
        // Test simple key replacement
        assertEquals("${ns.foo}", transformer.replaceKeys("${foo}"));

        // Test multiple key replacements
        assertEquals("${ns.foo} and ${ns.bar}", transformer.replaceKeys("${foo} and ${bar}"));

        // Test keys not in mapping
        assertEquals("${baz}", transformer.replaceKeys("${baz}"));

        // Test partial matches
        assertEquals("foobar", transformer.replaceKeys("foobar"));

        // Test PostMapping scenario
        assertEquals("@PostMapping({\"${wms-stock.schedule.clover.webAlias}/checkTaskTooManyAlarm\"})",
                transformer.replaceKeys("@PostMapping({\"${schedule.clover.webAlias}/checkTaskTooManyAlarm\"})"));

        // Test direct references (non ${} form)
        assertEquals("ns.foo.property", transformer.replaceKeys("foo.property"));

        // Test mixed scenario
        assertEquals("${ns.foo} is not ns.foo but is ${ns.foo}",
                transformer.replaceKeys("${foo} is not foo but is ${foo}"));

        // Test boundary cases
        assertEquals("a${ns.foo}b", transformer.replaceKeys("a${foo}b"));

        // Test incomplete placeholders
        assertEquals("{ns.foo}", transformer.replaceKeys("{foo}"));

        // Test complex expressions
        assertEquals("#{'${ns.sku.extendAttrs}'.split(',')}",
                transformer.replaceKeys("#{'${sku.extendAttrs}'.split(',')}"));
    }

    @Test
    void testTransformExpression() {
        // Test StringLiteralExpr
        assertEquals("\"${ns.foo}\"", transformExpression("\"${foo}\""));

        // Test StringLiteralExpr with multiple placeholders
        assertEquals("\"${ns.foo} and ${ns.bar}\"", transformExpression("\"${foo} and ${bar}\""));

        // Test StringLiteralExpr with text outside placeholders
        assertEquals("\"prefix ${ns.foo} suffix\"", transformExpression("\"prefix ${foo} suffix\""));

        // Test expression with no placeholders
        assertEquals("\"unchanged text\"", transformExpression("\"unchanged text\""));

        // Test complex expressions
        assertEquals("\"#{'${ns.sku.extendAttrs}'.split(',')}\"",
                transformExpression("\"#{'${sku.extendAttrs}'.split(',')}\""));
    }

    private String transformExpression(String expressionStr) {
        Expression expr = StaticJavaParser.parseExpression(expressionStr);
        Expression transformedExpr = transformer.transformAnnotationExpression(expr);
        return transformedExpr.toString();
    }

    @Test
    void testTransformAnnotation() {
        // Test @Value annotation
        testAnnotationTransformation("@Value(\"${foo}\")", "@Value(\"${ns.foo}\")");

        // Test @LafValue annotation
        testAnnotationTransformation("@LafValue(\"${foo}\")", "@LafValue(\"${ns.foo}\")");

        testAnnotationTransformation("@RequestMapping(\"/stockCheck\")", "@RequestMapping(\"/stockCheck\")");
        testAnnotationTransformation("@RequestMapping(\"/${foo}/stockCheck\")", "@RequestMapping(\"/${ns.foo}/stockCheck\")");

        // Test @ConfigurationProperties annotation
        testAnnotationTransformation("@ConfigurationProperties(prefix = \"foo\")",
                "@ConfigurationProperties(prefix = \"ns.foo\")");

        // Test @ConditionalOnProperty annotation
        testAnnotationTransformation("@ConditionalOnProperty(name = \"foo\", havingValue = \"true\")",
                "@ConditionalOnProperty(name = \"ns.foo\", havingValue = \"true\")");

        // Test @PostMapping annotation with array
        testAnnotationTransformation("@PostMapping({\"${schedule.clover.webAlias}/checkTaskTooManyAlarm\"})",
                "@PostMapping({ \"${wms-stock.schedule.clover.webAlias}/checkTaskTooManyAlarm\" })");

        // Test annotation that should not be modified
        testAnnotationTransformation("@Autowired", "@Autowired");

        // Unregistered annotation
        testAnnotationTransformation("@MyAnnotation(\"${foo}\")", "@MyAnnotation(\"${foo}\")");
        testAnnotationTransformation("@MyAnnotation(key = \"${foo}\", value = \"bar\")", "@MyAnnotation(key = \"${foo}\", value = \"bar\")");

        // Test @Value annotation with complex expression
        testAnnotationTransformation("@Value(\"#{'${sku.extendAttrs}'.split(',')}\")",
                "@Value(\"#{'${ns.sku.extendAttrs}'.split(',')}\")");

        // Test @GetMapping annotation
        testAnnotationTransformation("@GetMapping(\"/${foo}/endpoint\")",
                "@GetMapping(\"/${ns.foo}/endpoint\")");

        // Test annotation with multiple attributes
        testAnnotationTransformation("@ConditionalOnProperty(name = \"foo\", prefix = \"bar\", havingValue = \"true\")",
                "@ConditionalOnProperty(name = \"ns.foo\", prefix = \"ns.bar\", havingValue = \"true\")");
    }

    private void testAnnotationTransformation(String input, String expected) {
        AnnotationExpr annotationExpr = StaticJavaParser.parseAnnotation(input);
        transformer.transformAnnotation(annotationExpr);
        assertEquals(expected, annotationExpr.toString());
    }

    @Test
    void testEdgeCases() {
        // Test empty string
        assertEquals("", transformer.replaceKeys(""));

        // Test null key in mapping (should not happen, but testing for robustness)
        keyMapping.put(null, "some.value");
        assertEquals("${ns.foo}", transformer.replaceKeys("${foo}"));

        // Test key with special regex characters
        keyMapping.put("special.key.*+", "replaced.key");
        assertEquals("${replaced.key}", transformer.replaceKeys("${special.key.*+}"));

        // Test key that's a prefix of another key
        keyMapping.put("prefix", "new.prefix");
        keyMapping.put("prefix.extended", "new.extended");
        assertEquals("${new.prefix} ${new.extended}",
                transformer.replaceKeys("${prefix} ${prefix.extended}"));
    }
}
