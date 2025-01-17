package io.github.federate.extractor.ast;

public class ConcurrencyUsage {
    private String className;
    private String methodName;
    private String type; // "ThreadPool" 或 "NewThread"
    private String details;
    private int lineNumber;

    public ConcurrencyUsage(String className, String methodName, String type, String details, int lineNumber) {
        this.className = className;
        this.methodName = methodName;
        this.type = type;
        this.details = details;
        this.lineNumber = lineNumber;
    }
}
