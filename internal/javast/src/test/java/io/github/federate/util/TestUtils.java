package io.github.federate.util;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import io.github.federate.visitor.BaseCodeModifier;
import org.junit.jupiter.api.Assertions;

import java.io.IOException;

public class TestUtils {

    public static void assertTransformation(String testCaseName, BaseCodeModifier transformer, boolean shouldBeModified) throws IOException {
        TestCaseLoader.TestCase testCase = TestCaseLoader.load(testCaseName);
        CompilationUnit cu = StaticJavaParser.parse(testCase.getInput());
        transformer.visit(cu, (Void) null);

        if (shouldBeModified) {
            Assertions.assertTrue(transformer.isModified(), "Expected the transformer to modify the code.");
        }
        Assertions.assertEquals(testCase.getExpected().trim(), cu.toString().trim(), "The transformed code does not match the expected output.");
    }

    public static void assertTransformation(String testCaseName, BaseCodeModifier transformer) throws IOException {
        assertTransformation(testCaseName, transformer, true);
    }
}