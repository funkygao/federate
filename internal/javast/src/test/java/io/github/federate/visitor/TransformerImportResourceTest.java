package io.github.federate.visitor;

import io.github.federate.util.TestUtils;
import org.junit.jupiter.api.Test;

import java.io.IOException;

class TransformerImportResourceTest {

    @Test
    void testSingleResourceTransformation() throws IOException {
        TestUtils.assertTransformation("import_resource_single", new TransformerImportResource("testComponent"));
    }

    @Test
    void testSingleResourceWithoutClasspathTransformation() throws IOException {
        TestUtils.assertTransformation("import_resource_without_classpath", new TransformerImportResource("testComponent"));
    }

    @Test
    void testMultipleResourcesTransformation() throws IOException {
        TestUtils.assertTransformation("import_resource_multiple", new TransformerImportResource("testComponent"));
    }

    @Test
    void testLocationsAttributeTransformation() throws IOException {
        TestUtils.assertTransformation("import_resource_locations", new TransformerImportResource("testComponent"));
    }

    @Test
    void testMixedResourcesTransformation() throws IOException {
        TestUtils.assertTransformation("import_resource_mixed", new TransformerImportResource("testComponent"));
    }

    @Test
    void testNoTransformationNeeded() throws IOException {
        TestUtils.assertTransformation("import_resource_no_transform", new TransformerImportResource("testComponent"), false);
    }

    @Test
    void testComplexClassWithMultipleAnnotations() throws IOException {
        TestUtils.assertTransformation("import_resource_complex", new TransformerImportResource("testComponent"));
    }
}