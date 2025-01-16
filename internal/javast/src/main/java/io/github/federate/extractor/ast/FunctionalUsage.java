package io.github.federate.extractor.ast;

public class FunctionalUsage {
    String methodName;
    String fileName;
    int lineNumber;
    String type; // "lambda" 或 "stream"
    String operation; // 例如 "map", "filter", "reduce" 等
    String context; // 使用的上下文，例如方法名或变量名

    public FunctionalUsage(String methodName, String fileName, int lineNumber, String type, String operation, String context) {
        this.methodName = methodName;
        this.fileName = fileName;
        this.lineNumber = lineNumber;
        this.type = type;
        this.operation = operation;
        this.context = context;
    }
}
