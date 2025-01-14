package io.github.federate.extractor.info;

public class ComplexLoop {
    String methodName;
    String fileName;
    String loopType;
    int lineNumber;
    int nestingLevel;
    int bodySize;

    public ComplexLoop(String methodName, String fileName, String loopType, int lineNumber, int nestingLevel, int bodySize) {
        this.methodName = methodName;
        this.fileName = fileName;
        this.loopType = loopType;
        this.lineNumber = lineNumber;
        this.nestingLevel = nestingLevel;
        this.bodySize = bodySize;
    }
}
