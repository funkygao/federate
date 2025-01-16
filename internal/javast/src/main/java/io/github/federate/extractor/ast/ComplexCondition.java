package io.github.federate.extractor.ast;

public class ComplexCondition {
    String fileName;
    String methodName;
    String condition;
    int complexity;
    int lineNumber;

    public ComplexCondition(String fileName, String methodName, String condition, int complexity, int lineNumber) {
        this.fileName = fileName;
        this.methodName = methodName;
        this.condition = condition;
        this.complexity = complexity;
        this.lineNumber = lineNumber;
    }
}
