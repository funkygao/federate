package io.github.federate.visitor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.AnnotationExpr;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TransformerImportResourceTest {

    @Test
    void testSingleResourceTransformation() {
        String input = "@ImportResource(\"classpath:config.xml\")";
        String expected = "@ImportResource(\"classpath:federated/testComponent/config.xml\")";
        assertTransformation(input, expected);
    }

    @Test
    void testSingleResourceWithoutClasspathTransformation() {
        String input = "@ImportResource(\"/foo/config.xml\")";
        String expected = "@ImportResource(\"/federated/testComponent/foo/config.xml\")";
        assertTransformation(input, expected);
    }

    @Test
    void testMultipleResourcesTransformation() {
        String input = "@ImportResource({\"config1.xml\", \"config2.xml\"})";
        String expected = "@ImportResource({ \"federated/testComponent/config1.xml\", \"federated/testComponent/config2.xml\" })";
        assertTransformation(input, expected);
    }

    @Test
    void testLocationsAttributeTransformation() {
        String input = "@ImportResource(locations = {\"classpath:config1.xml\", \"classpath:config2.xml\"})";
        String expected = "@ImportResource(locations = { \"classpath:federated/testComponent/config1.xml\", \"classpath:federated/testComponent/config2.xml\" })";
        assertTransformation(input, expected);
    }

    @Test
    void testMixedResourcesTransformation() {
        String input = "@ImportResource({\"classpath:config1.xml\", \"file:config2.xml\"})";
        String expected = "@ImportResource({ \"classpath:federated/testComponent/config1.xml\", \"federated/testComponent/file:config2.xml\" })";
        assertTransformation(input, expected);
    }

    @Test
    void testNoTransformationNeeded() {
        String input = "@ImportResource(\"/federated/testComponent/config.xml\")";
        assertTransformation(input, input, false);
    }

    @Test
    void testComplexClassWithMultipleAnnotations() {
        String input = "@ImportResource({\"classpath:applicationContext-common.xml\", \"classpath:applicationContext-beans.xml\"})\n" +
                "@PropertySource(\"classpath:foo.properties\")\n" +
                "public class AppConfig {\n" +
                "    // Some code here\n" +
                "}";
        String expected = "@ImportResource({ \"classpath:federated/testComponent/applicationContext-common.xml\", \"classpath:federated/testComponent/applicationContext-beans.xml\" })";
        assertTransformation(input, expected);
    }

    private void assertTransformation(String input, String expected, boolean modified) {
        CompilationUnit cu = StaticJavaParser.parse("public class Test { " + input + " void method() {} }");
        TransformerImportResource transformer = new TransformerImportResource("testComponent");
        transformer.visit(cu, (Void) null);  // Explicitly cast to Void to resolve ambiguity
        Optional<AnnotationExpr> transformedAnnotationOpt = cu.findFirst(AnnotationExpr.class);
        assertTrue(transformedAnnotationOpt.isPresent(), "Annotation not found");
        assertEquals(expected, transformedAnnotationOpt.get().toString());
        if (modified) {
            assertTrue(transformer.isModified());
        }
    }

    private void assertTransformation(String input, String expected) {
        assertTransformation(input, expected, true);
    }
}
