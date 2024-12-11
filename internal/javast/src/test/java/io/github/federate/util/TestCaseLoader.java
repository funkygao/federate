package io.github.federate.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

public class TestCaseLoader {
    private static final String TEST_RESOURCES_PATH = "transformer/";
    private static final String INPUT_DELIMITER = "// INPUT";
    private static final String EXPECTED_DELIMITER = "// EXPECTED";

    public static class TestCase {
        private final String input;
        private final String expected;

        public TestCase(String input, String expected) {
            this.input = input;
            this.expected = expected;
        }

        public String getInput() {
            return input;
        }

        public String getExpected() {
            return expected;
        }
    }

    public static TestCase load(String testCaseName) throws IOException {
        String resourcePath = TEST_RESOURCES_PATH + testCaseName + ".java";
        InputStream is = TestCaseLoader.class.getClassLoader().getResourceAsStream(resourcePath);
        if (is == null) {
            throw new IOException("Could not find test case file: " + resourcePath);
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(is, StandardCharsets.UTF_8))) {

            // 使用 Java 8 的 Stream API 读取文件内容
            String content = reader.lines().collect(Collectors.joining("\n"));

            String[] parts = content.split(EXPECTED_DELIMITER);
            if (parts.length != 2) {
                throw new IllegalStateException("Invalid test case file format: " + resourcePath);
            }

            String input = parts[0].replace(INPUT_DELIMITER, "").trim();
            String expected = parts[1].trim();

            return new TestCase(input, expected);
        }
    }
}