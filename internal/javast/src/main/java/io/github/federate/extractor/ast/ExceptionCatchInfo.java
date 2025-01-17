package io.github.federate.extractor.ast;

import java.util.List;

public class ExceptionCatchInfo {
    private String className;
    private String methodName;
    private List<String> exceptionTypes;
    private int lineNumber;

    public ExceptionCatchInfo(String className, String methodName, List<String> exceptionTypes, int lineNumber) {
        this.className = className;
        this.methodName = methodName;
        this.exceptionTypes = exceptionTypes;
        this.lineNumber = lineNumber;
    }
}
