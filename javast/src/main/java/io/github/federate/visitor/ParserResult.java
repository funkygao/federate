package io.github.federate.visitor;

import java.util.List;

public class ParserResult {
    private String filePath;
    private List<String> classes;
    private List<String> methods;

    public ParserResult(String filePath, List<String> classes, List<String> methods) {
        this.filePath = filePath;
        this.classes = classes;
        this.methods = methods;
    }

    // Getters and setters
}

