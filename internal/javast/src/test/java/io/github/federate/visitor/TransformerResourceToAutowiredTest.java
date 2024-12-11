package io.github.federate.visitor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import io.github.federate.util.TestCaseLoader;
import io.github.federate.util.TestCaseLoader.TestCase;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TransformerResourceToAutowiredTest {

    @Test
    void resourceWithNameAndSetterMethod() throws IOException {
        assertTransformation("resource_with_name_setter_method");
    }

    @Test
    void testBasicTransformation() throws IOException {
        assertTransformation("basic_transformation");
    }

    @Test
    void testMultipleSameTypeFields() throws IOException {
        assertTransformation("multiple_same_type_fields");
    }

    @Test
    void testSetterMethodTransformation() throws IOException {
        assertTransformation("setter_method_transformation");
    }

    @Test
    void testWithOtherAnnotations() throws IOException {
        assertTransformation("with_other_annotations");
    }

    @Test
    void testGenericTypes() throws IOException {
        assertTransformation("generic_types");
    }

    @Test
    void testInnerClass() throws IOException {
        assertTransformation("inner_class");
    }

    private void assertTransformation(String testCaseName) throws IOException {
        TestCase testCase = TestCaseLoader.load(testCaseName);
        CompilationUnit cu = StaticJavaParser.parse(testCase.getInput());
        TransformerResourceToAutowired transformer = new TransformerResourceToAutowired();
        transformer.visit(cu, (Void) null);

        assertTrue(transformer.isModified());
        assertEquals(testCase.getExpected(), cu.toString().trim());
    }
}