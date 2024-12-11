package io.github.federate.visitor;

import io.github.federate.util.TestUtils;
import org.junit.jupiter.api.Test;

import java.io.IOException;

class TransformerResourceToAutowiredTest {

    @Test
    void testBasicTransformation() throws IOException {
        TestUtils.assertTransformation("basic_transformation", new TransformerResourceToAutowired());
    }

    @Test
    void testMultipleSameTypeFields() throws IOException {
        TestUtils.assertTransformation("multiple_same_type_fields", new TransformerResourceToAutowired());
    }

    @Test
    void testSetterMethodTransformation() throws IOException {
        TestUtils.assertTransformation("setter_method_transformation", new TransformerResourceToAutowired());
    }

    @Test
    void testWithOtherAnnotations() throws IOException {
        TestUtils.assertTransformation("with_other_annotations", new TransformerResourceToAutowired());
    }

    @Test
    void testGenericTypes() throws IOException {
        TestUtils.assertTransformation("generic_types", new TransformerResourceToAutowired());
    }

    @Test
    void testInnerClass() throws IOException {
        TestUtils.assertTransformation("inner_class", new TransformerResourceToAutowired());
    }

    @Test
    void resourceWithNameAndSetterMethod() throws IOException {
        TestUtils.assertTransformation("resource_with_name_setter_method", new TransformerResourceToAutowired());
    }
}